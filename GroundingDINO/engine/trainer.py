import os
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


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.is_distributed = cfg.get('DISTRIBUTED', False)
        self.local_rank = cfg.get('LOCAL_RANK', -1)
        self.global_rank = 0
        self.world_size = 1

        if self.is_distributed:
            dist.init_process_group(
                backend='nccl',
                init_method=f'tcp://{cfg.MASTER_ADDR}:{cfg.MASTER_PORT}',
                rank=self.local_rank,
                world_size=self.cfg.get('WORLD_SIZE', 1)
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
            prefetch_factor=2
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

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.SOLVER.EPOCHS,
            eta_min=1e-6
        )

        self.grad_accum_steps = cfg.SOLVER.get('GRADIENT_ACCUMULATION_STEPS', 1)
        self.best_loss = float('inf')
        self.early_stopping_counter = 0

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cfg.LOG_DIR, exist_ok=True)

        self.log_file = os.path.join(cfg.LOG_DIR, 'training.log')
        if self.is_main_process:
            with open(self.log_file, 'w') as f:
                f.write(f"Training started at {datetime.now()}\n")
                f.write(f"Config: {cfg}\n")
                f.write(f"Device: {self.device}\n")
                f.write(f"Distributed: {self.is_distributed}, World Size: {self.world_size}\n")
                f.write(f"AMP: {self.amp}\n")
                f.write(f"Gradient Accumulation: {self.grad_accum_steps}\n")
                f.write("="*50 + "\n")

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
                progress_bar.set_postfix({'loss': loss.item() * self.grad_accum_steps})

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

        if self.is_distributed:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            num_batches_tensor = torch.tensor(num_batches, device=self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
            total_loss = total_loss_tensor.item()
            num_batches = num_batches_tensor.item()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def train(self):
        self.log(f"Training started on {self.device}")
        self.log(f"Total epochs: {self.cfg.SOLVER.EPOCHS}, Batch size: {self.cfg.SOLVER.BATCH_SIZE}")
        self.log(f"Effective batch size: {self.cfg.SOLVER.BATCH_SIZE * self.world_size * self.grad_accum_steps}")

        for epoch in range(self.cfg.SOLVER.EPOCHS):
            train_loss = self.train_epoch(epoch)
            
            if self.is_main_process:
                self.log(f'Epoch {epoch}: Train Loss = {train_loss:.4f}')

            if (epoch + 1) % self.cfg.TEST.EVAL_PERIOD == 0:
                val_loss = self.validate()
                
                if self.is_main_process:
                    self.log(f'Epoch {epoch}: Val Loss = {val_loss:.4f}')

                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.early_stopping_counter = 0
                        self.save_checkpoint(epoch, val_loss)
                        self.log(f'New best model saved!')
                    else:
                        self.early_stopping_counter += self.cfg.TEST.EVAL_PERIOD
                        self.log(f'Early stopping counter: {self.early_stopping_counter}/{self.cfg.TEST.get("EARLY_STOPPING_PATIENCE", 15)}')

                        if self.early_stopping_counter >= self.cfg.TEST.get("EARLY_STOPPING_PATIENCE", 15):
                            self.log(f'Early stopping triggered after {epoch+1} epochs')
                            break

            self.scheduler.step()

        if self.is_main_process:
            self.log(f'Training completed. Best validation loss: {self.best_loss:.4f}')
        
        if self.is_distributed:
            dist.destroy_process_group()

    def save_checkpoint(self, epoch, val_loss):
        if not self.is_main_process:
            return
        
        path = os.path.join(
            self.cfg.OUTPUT_DIR,
            f'grounding_dino_multi_modal_epoch_{epoch}_loss_{val_loss:.4f}.pth'
        )
        
        model_state = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'amp_scaler': self.scaler.state_dict() if self.scaler else None
        }, path)
        self.log(f'Checkpoint saved to {path}')

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        
        model_state = checkpoint['model_state_dict']
        if self.is_distributed and 'module.' not in list(model_state.keys())[0]:
            model_state = {f'module.{k}': v for k, v in model_state.items()}
        
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint['best_loss']
        
        if self.scaler and 'amp_scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['amp_scaler'])
        
        self.log(f'Checkpoint loaded from {path}')