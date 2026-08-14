import os
import signal
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from datetime import datetime

from groundingdino.datasets.dataset import MultiModalDataset, MultiModalCollator
from groundingdino.models.groundingdino import GroundingDINO
from groundingdino.engine.metrics import MetricsLogger, compute_map


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.is_distributed = cfg.DISTRIBUTED.ENABLED
        self.local_rank = cfg.LOCAL_RANK
        self.global_rank = 0
        self.world_size = 1

        self._paused = False
        self._stop_requested = False
        self._resume_epoch = 0
        self.pause_flag_file = os.path.join(cfg.OUTPUT_DIR, 'pause.flag')
        self.resume_flag_file = os.path.join(cfg.OUTPUT_DIR, 'resume.flag')
        self.stop_flag_file = os.path.join(cfg.OUTPUT_DIR, 'stop.flag')

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        if self.is_distributed:
            dist.init_process_group(
                backend='nccl',
                init_method=f'tcp://{cfg.MASTER_ADDR}:{cfg.MASTER_PORT}',
                rank=self.local_rank,
                world_size=self.cfg.DISTRIBUTED.WORLD_SIZE
            )
            self.global_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
            self.is_main_process = self.global_rank == 0
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.is_main_process = True

        self.model = GroundingDINO(cfg).to(self.device)

        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)

        self.amp = cfg.SOLVER.get('AMP', False)
        self.scaler = torch.amp.GradScaler('cuda') if self.amp and self.device.type == 'cuda' else None

        self.train_dataset = MultiModalDataset(cfg, split='train')
        self.val_dataset = MultiModalDataset(cfg, split='val')

        self.collator = MultiModalCollator(cfg)

        if self.is_distributed:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None

        self.num_workers = cfg.SOLVER.get('NUM_WORKERS', min(os.cpu_count(), 4))

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=(self.train_sampler is None),
            collate_fn=self.collator,
            num_workers=self.num_workers,
            sampler=self.train_sampler,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            sampler=self.val_sampler,
            pin_memory=True
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )

        self.base_lr = cfg.SOLVER.BASE_LR
        self.warmup_epochs = cfg.SOLVER.get('WARMUP_EPOCHS', 0)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, cfg.SOLVER.EPOCHS - self.warmup_epochs),
            eta_min=1e-6
        )

        self.grad_accum_steps = cfg.SOLVER.get('GRADIENT_ACCUMULATION_STEPS', 1)
        self.best_loss = float('inf')
        self.early_stopping_counter = 0

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cfg.LOG_DIR, exist_ok=True)

        self.log_file = os.path.join(cfg.LOG_DIR, 'training.log')
        self.metrics_logger = MetricsLogger(cfg.LOG_DIR) if self.is_main_process else None
        if self.is_main_process:
            with open(self.log_file, 'w') as f:
                f.write(f"Training started at {datetime.now()}\n")
                f.write(f"Config: {cfg}\n")
                f.write(f"Device: {self.device}\n")
                f.write(f"Distributed: {self.is_distributed}, World Size: {self.world_size}\n")
                f.write(f"AMP: {self.amp}\n")
                f.write(f"Gradient Accumulation: {self.grad_accum_steps}\n")
                f.write("="*50 + "\n")

    def _signal_handler(self, signum, frame):
        self._stop_requested = True
        self.log(f"\n[Signal] Received signal {signum}, will stop after current epoch...")

    def _check_flags(self):
        # 暂停: 创建 pause.flag → 当前 epoch 结束后暂停
        if not self._paused and os.path.exists(self.pause_flag_file):
            self._paused = True
            self.log(f"\n[Pause] Pause flag detected. Training will pause after this epoch.")
            os.remove(self.pause_flag_file)

        # 恢复: 暂停期间创建 resume.flag → 恢复训练
        if self._paused and os.path.exists(self.resume_flag_file):
            self._paused = False
            self.log(f"\n[Resume] Resume flag detected. Training will resume.")
            os.remove(self.resume_flag_file)

        # 停止: 创建 stop.flag → 当前 epoch 结束后停止
        if os.path.exists(self.stop_flag_file):
            self._stop_requested = True
            self.log(f"\n[Stop] Stop flag detected. Training will stop after this epoch.")
            os.remove(self.stop_flag_file)

    def _auto_save(self, epoch, train_loss):
        if self.is_main_process and epoch % 5 == 0:
            self.save_checkpoint(epoch, train_loss, is_latest=True)

    def log(self, message):
        print(message)
        if self.is_main_process:
            with open(self.log_file, 'a') as f:
                f.write(f"{datetime.now()} - {message}\n")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        batch_idx = 0

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}', disable=not self.is_main_process)
        for batch in progress_bar:
            inputs = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(inputs)
                    loss = outputs['loss'] / self.grad_accum_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(inputs)
                loss = outputs['loss'] / self.grad_accum_steps
                loss.backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            batch_idx += 1

            if self.is_main_process:
                batch_loss = loss.item() * self.grad_accum_steps
                progress_bar.set_postfix({'loss': batch_loss})
                loss_dict = outputs.get('loss_dict') if isinstance(outputs, dict) else None
                self.metrics_logger.log_batch(
                    epoch, batch_idx - 1,
                    self.optimizer.param_groups[0]['lr'],
                    batch_loss, loss_dict
                )

        if self.is_distributed:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            num_batches_tensor = torch.tensor(num_batches, device=self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
            total_loss = total_loss_tensor.item()
            num_batches = num_batches_tensor.item()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_gts = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation', disable=not self.is_main_process):
                inputs = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                if self.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                if 'loss' in outputs:
                    total_loss += outputs['loss'].item()
                    num_batches += 1

                # 收集预测框和 GT 用于 mAP 计算 (cxcywh 归一化格式)
                if 'pred_boxes' in outputs and 'pred_logits' in outputs:
                    pred_boxes = outputs['pred_boxes']          # [B, N, 4]
                    scores = torch.sigmoid(outputs['pred_logits']).max(dim=-1)[0]  # [B, N]
                    for b in range(pred_boxes.shape[0]):
                        all_preds.append({
                            'boxes': pred_boxes[b].cpu(),
                            'scores': scores[b].cpu(),
                        })
                for t in inputs.get('targets', []):
                    all_gts.append({'boxes': t['boxes'].cpu() if isinstance(t['boxes'], torch.Tensor) else t['boxes']})

        if self.is_distributed:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            num_batches_tensor = torch.tensor(num_batches, device=self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
            total_loss = total_loss_tensor.item()
            num_batches = num_batches_tensor.item()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        mAP50 = compute_map(all_preds, all_gts, iou_threshold=0.5)
        return avg_loss, mAP50

    def train(self, resume_from=None):
        if resume_from:
            self.load_checkpoint(resume_from, load_training_state=True)
            start_epoch = self._resume_epoch + 1
            self.log(f"Resuming training from epoch {start_epoch}")
        else:
            start_epoch = 0

        self.log(f"Training started on {self.device}")
        self.log(f"Total epochs: {self.cfg.SOLVER.EPOCHS}, Batch size: {self.cfg.SOLVER.BATCH_SIZE}")
        self.log(f"Effective batch size: {self.cfg.SOLVER.BATCH_SIZE * self.world_size * self.grad_accum_steps}")
        self.log(f"Pause/Resume/Stop: python tools/pause_training.py {{pause|resume|stop|status}}")

        for epoch in range(start_epoch, self.cfg.SOLVER.EPOCHS):
            self._check_flags()

            if self._stop_requested:
                self.log(f'\n[Stop] Training stopped by user request at epoch {epoch}')
                self.save_checkpoint(epoch, 0, is_latest=True)
                break

            if self._paused:
                self.log(f'\n[Pause] Training paused at epoch {epoch}')
                self.save_checkpoint(epoch, 0, is_latest=True)
                self.log(f'[Pause] Waiting for resume...')
                self.log(f'[Pause] Run: python tools/pause_training.py resume')
                self.log(f'[Pause] Or stop: python tools/pause_training.py stop')
                
                while self._paused:
                    time.sleep(1)
                    self._check_flags()
                    
                    if self._stop_requested:
                        self.log(f'\n[Stop] Stop requested while paused. Exiting.')
                        break
                
                if self._stop_requested:
                    break
                
                self.log(f'[Pause] Training resumed at epoch {epoch}')

            if epoch < self.warmup_epochs:
                warmup_lr = self.base_lr * (epoch + 1) / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg['lr'] = warmup_lr
                if self.is_main_process:
                    self.log(f'[Warmup] Epoch {epoch}: lr = {warmup_lr:.6f}')

            epoch_start = time.time()
            train_loss = self.train_epoch(epoch)
            epoch_elapsed = time.time() - epoch_start

            if self.is_main_process:
                self.log(f'Epoch {epoch}: Train Loss = {train_loss:.4f}  ({epoch_elapsed:.1f}s)')

            self._auto_save(epoch, train_loss)

            if (epoch + 1) % self.cfg.TEST.EVAL_PERIOD == 0:
                val_loss, mAP50 = self.validate()

                if self.is_main_process:
                    self.log(f'Epoch {epoch}: Val Loss = {val_loss:.4f}  mAP@50 = {mAP50:.4f}')
                    self.metrics_logger.log_epoch(
                        epoch, train_loss, val_loss, mAP50,
                        self.optimizer.param_groups[0]['lr'], epoch_elapsed
                    )

                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.early_stopping_counter = 0
                        self.save_checkpoint(epoch, val_loss)
                        self.log(f'New best model saved! (mAP@50={mAP50:.4f})')
                    else:
                        self.early_stopping_counter += self.cfg.TEST.EVAL_PERIOD
                        self.log(f'Early stopping counter: {self.early_stopping_counter}/{self.cfg.TEST.get("EARLY_STOPPING_PATIENCE", 15)}')

                        if self.early_stopping_counter >= self.cfg.TEST.get("EARLY_STOPPING_PATIENCE", 15):
                            self.log(f'Early stopping triggered after {epoch+1} epochs')
                            self.save_checkpoint(epoch, val_loss, is_latest=True)
                            break

            if epoch >= self.warmup_epochs:
                self.scheduler.step()

        if self.is_main_process:
            self.log(f'Training completed. Best validation loss: {self.best_loss:.4f}')
        
        if self.is_distributed:
            dist.destroy_process_group()

    def save_checkpoint(self, epoch, val_loss, is_latest=False):
        if not self.is_main_process:
            return
        
        if is_latest:
            path = os.path.join(self.cfg.OUTPUT_DIR, 'latest_checkpoint.pth')
        else:
            path = os.path.join(
                self.cfg.OUTPUT_DIR,
                f'grounding_dino_multi_modal_epoch_{epoch}_loss_{val_loss:.4f}.pth'
            )
        
        model_state = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'amp_scaler': self.scaler.state_dict() if self.scaler else None,
            'early_stopping_counter': self.early_stopping_counter,
            'config': {
                'EPOCHS': self.cfg.SOLVER.EPOCHS,
                'BASE_LR': self.cfg.SOLVER.BASE_LR,
                'WEIGHT_DECAY': self.cfg.SOLVER.WEIGHT_DECAY,
            }
        }
        
        torch.save(checkpoint_data, path)
        
        if is_latest:
            self.log(f'[Auto-save] Latest checkpoint saved to {path} (epoch {epoch})')
        else:
            self.log(f'Checkpoint saved to {path}')

    def load_checkpoint(self, path, load_training_state=True):
        checkpoint = torch.load(path, map_location=self.device)
        
        model_state = checkpoint['model_state_dict']
        if self.is_distributed and 'module.' not in list(model_state.keys())[0]:
            model_state = {f'module.{k}': v for k, v in model_state.items()}
        
        self.model.load_state_dict(model_state)
        
        if load_training_state:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
            self._resume_epoch = checkpoint.get('epoch', -1)
            
            if self.scaler and 'amp_scaler' in checkpoint and checkpoint['amp_scaler']:
                self.scaler.load_state_dict(checkpoint['amp_scaler'])
            
            self.log(f'Checkpoint loaded from {path} (resume from epoch {self._resume_epoch})')
        else:
            self.log(f'Model weights loaded from {path} (inference only)')