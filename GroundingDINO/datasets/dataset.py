import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
import cv2
import random

try:
    from transformers import BertTokenizerFast
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class MultiModalDataset(Dataset):
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split
        self.data_path = cfg.DATASETS.DATA_PATH
        self.modalities = cfg.MODEL.MULTI_MODAL.MODALITIES

        self.tokenizer = self._create_tokenizer(cfg.MODEL.TEXT_ENCODER.NAME)
        self.max_seq_len = cfg.MODEL.get('MAX_SEQ_LEN', 256)
        
        # 训练时启用数据增强
        self.is_train = (split == 'train')
        self.aug_prob = 0.5  # 增强概率

        self.data = self._load_data()

        if split == 'train':
            max_samples = cfg.DATASETS.get('MAX_TRAIN_SAMPLES', -1)
        else:
            max_samples = cfg.DATASETS.get('MAX_VAL_SAMPLES', -1)
        if max_samples is not None and max_samples > 0:
            self.data = self.data[:max_samples]
            print(f"[Dataset] {split} split limited to first {len(self.data)} samples")
        
        if self.is_train:
            print(f"[Dataset] Training mode: data augmentation enabled (prob={self.aug_prob})")

    def _create_tokenizer(self, model_name):
        if HAS_TRANSFORMERS:
            try:
                tokenizer = BertTokenizerFast.from_pretrained(model_name, local_files_only=True)
                print(f"[Dataset] Loaded BertTokenizerFast from local cache: {model_name}")
                return tokenizer
            except Exception:
                pass

            try:
                tokenizer = BertTokenizerFast.from_pretrained(model_name)
                print(f"[Dataset] Loaded BertTokenizerFast: {model_name}")
                return tokenizer
            except Exception as e:
                print(f"[Dataset] Cannot load tokenizer ({e}), using fallback")

        return SimpleTokenizer()

    def _load_data(self):
        # 自定义文件名优先
        custom_field = 'TRAIN_FILE' if self.split == 'train' else 'VAL_FILE'
        custom_name = self.cfg.DATASETS.get(custom_field, None) if hasattr(self.cfg.DATASETS, custom_field) else None
        if custom_name:
            ann_file = custom_name if os.path.isabs(custom_name) else os.path.join(self.data_path, custom_name)
        else:
            ann_file = os.path.join(self.data_path, f'{self.split}.json')

        if not os.path.exists(ann_file):
            # 尝试 fused + backup fallback
            for candidate in [
                os.path.join(self.data_path, f'{self.split}_fused.json'),
                os.path.join(self.data_path, f'{self.split}_backup.json'),
            ]:
                if os.path.exists(candidate):
                    ann_file = candidate
                    break
            else:
                print(f"[Dataset] {self.split} ann file missing -> returning empty")
                return []
        print(f"[Dataset] {self.split} loaded from: {ann_file}")
        with open(ann_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        target_size = (512, 512)

        orig_size = None
        size_fb = item.get('_orig_size_fallback') or None
        if size_fb is None:
            anns = item.get('annotations') or []
            if len(anns) > 0 and 'img_size' in (anns[0] or {}):
                size_fb = anns[0]['img_size']

        inputs = {}
        for modality in self.modalities:
            img = None
            size = None
            img_path = item.get(modality)
            if img_path:
                if not os.path.isabs(img_path):
                    img_path = os.path.join(self.data_path, img_path)
                if os.path.exists(img_path):
                    try:
                        img, size = self._load_image(img_path, modality, return_size=True)
                    except Exception:
                        img = None
                        size = None
            if img is None:
                fallback_w, fallback_h = size_fb or target_size
                ch = 3 if modality == 'rgb' else 1
                img = torch.zeros(ch, *target_size)
                size = (int(fallback_w), int(fallback_h))
                inputs['_' + modality + '_fallback'] = True
            inputs[modality] = img
            if orig_size is None:
                orig_size = size

        if orig_size is None and size_fb is not None:
            orig_size = (int(size_fb[0]), int(size_fb[1]))
        if orig_size is None:
            orig_size = target_size

        text = item.get('text', item.get('query', item.get('prompt', 'target')))
        text_input_ids, text_attention_mask = self._tokenize(text)
        inputs['text'] = text
        inputs['text_input_ids'] = text_input_ids
        inputs['text_attention_mask'] = text_attention_mask

        if 'annotations' in item and orig_size is not None:
            targets = self._prepare_targets(item['annotations'], orig_size)
            inputs['targets'] = targets

        if self.is_train and random.random() < self.aug_prob:
            inputs = self._augment(inputs)

        return inputs

    def _augment(self, inputs):
        """
        数据增强 - 重点：通过几何变换打破位置记忆，迫使模型学习特征→位置的映射
        - 随机水平/垂直翻转
        - 随机缩放（0.7~1.3倍）+ 边缘pad或裁剪，改变目标相对位置和尺寸
        - 随机平移（±15%），将目标移动到图像任意区域
        - 亮度 / 对比度微调（仅外观，不影响位置）
        """
        aug_inputs = dict(inputs)
        flip_h = random.random() < 0.5
        flip_v = random.random() < 0.3

        scale = random.uniform(0.7, 1.3)
        shift_x = random.uniform(-0.15, 0.15)
        shift_y = random.uniform(-0.15, 0.15)

        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)

        for mod in self.modalities:
            if mod not in aug_inputs or not isinstance(aug_inputs[mod], torch.Tensor):
                continue
            img = aug_inputs[mod]

            if flip_h:
                img = torch.flip(img, dims=[2])
            if flip_v:
                img = torch.flip(img, dims=[1])

            C, H, W = img.shape
            img_np = img.permute(1, 2, 0).numpy()
            M = np.float32([[scale, 0, (W * (1 - scale) / 2) + (shift_x * W)],
                            [0, scale, (H * (1 - scale) / 2) + (shift_y * H)]])
            flags = cv2.INTER_LINEAR if mod == 'rgb' else cv2.INTER_NEAREST
            img_np = cv2.warpAffine(img_np, M, (W, H), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if img_np.ndim == 2:
                img_np = img_np[..., None]
            img = torch.from_numpy(img_np).permute(2, 0, 1).float()

            if mod == 'rgb':
                img = torch.clamp(img * brightness, 0.0, 1.0)
                mean = img.mean()
                img = torch.clamp((img - mean) * contrast + mean, 0.0, 1.0)
            elif mod in ('ir', 'depth'):
                img = torch.clamp(img * brightness, 0.0, 1.0)

            aug_inputs[mod] = img

        if 'targets' in aug_inputs:
            targets = aug_inputs['targets']
            boxes = targets['boxes'].clone()
            if flip_h:
                boxes[:, 0] = 1.0 - boxes[:, 0]
            if flip_v:
                boxes[:, 1] = 1.0 - boxes[:, 1]
            boxes[:, 0] = boxes[:, 0] * scale + shift_x
            boxes[:, 1] = boxes[:, 1] * scale + shift_y
            boxes[:, 2] = boxes[:, 2] * scale
            boxes[:, 3] = boxes[:, 3] * scale
            boxes[:, 0] = torch.clamp(boxes[:, 0], 0.0, 1.0)
            boxes[:, 1] = torch.clamp(boxes[:, 1], 0.0, 1.0)
            boxes[:, 2] = torch.clamp(boxes[:, 2], 0.01, 1.0)
            boxes[:, 3] = torch.clamp(boxes[:, 3], 0.01, 1.0)
            targets['boxes'] = boxes
            aug_inputs['targets'] = targets

        return aug_inputs

    def _load_image(self, img_path, modality, return_size=False):
        target_size = (512, 512)

        if modality == 'depth':
            img = None
            try:
                pil_img = Image.open(img_path)
                img = np.array(pil_img)
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            except Exception:
                pass

            if img is None:
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if img is None:
                return torch.zeros(1, *target_size) if not return_size else (torch.zeros(1, *target_size), target_size)

            orig_h, orig_w = img.shape[:2]
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
            if img.dtype == np.uint16:
                img = (img / 65535.0).astype(np.float32)
            elif img.dtype == np.int32:
                img = (img / 65535.0).astype(np.float32)
            else:
                img = img.astype(np.float32) / 255.0
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            else:
                img = img.transpose(2, 0, 1)
        elif modality == 'ir':
            img = Image.open(img_path).convert('L')
            orig_w, orig_h = img.size
            img = img.resize(target_size, Image.BILINEAR)
            img = np.array(img).astype(np.float32) / 255.0
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
        else:
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            img = img.resize(target_size, Image.BILINEAR)
            img = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0

        tensor = torch.from_numpy(img.copy())
        if return_size:
            return tensor, (orig_w, orig_h)
        return tensor

    def _tokenize(self, text):
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors='pt'
        )
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)

    def _prepare_targets(self, annotations, orig_size):
        orig_w, orig_h = orig_size
        labels = []
        boxes = []
        for ann in annotations:
            labels.append(ann['category_id'])
            x, y, w, h = ann['bbox']
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            nw = w / orig_w
            nh = h / orig_h
            boxes.append([cx, cy, nw, nh])
        return {'labels': torch.tensor(labels, dtype=torch.long),
                'boxes': torch.tensor(boxes, dtype=torch.float32)}


class SimpleTokenizer:
    def __init__(self, vocab_size=30522):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.mask_token_id = 3

        self._char_to_id = {}
        self._id_to_char = {}
        chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?;:\'"()[]{}<>-=_+*/\\|@#$%^&~`'

        idx = 4
        for c in chars:
            self._char_to_id[c] = idx
            self._id_to_char[idx] = c
            idx += 1

        self.unk_token_id = idx

    def encode(self, text, padding='max_length', truncation=True, max_length=256, return_tensors='pt'):
        tokens = [self.cls_token_id]

        for c in text:
            if len(tokens) >= max_length - 1:
                break
            tokens.append(self._char_to_id.get(c, self.unk_token_id))

        tokens.append(self.sep_token_id)

        attention_mask = [1] * len(tokens)

        if padding == 'max_length' and len(tokens) < max_length:
            padding_len = max_length - len(tokens)
            tokens.extend([self.pad_token_id] * padding_len)
            attention_mask.extend([0] * padding_len)

        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        if return_tensors == 'pt':
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def __call__(self, text, padding='max_length', truncation=True, max_length=256, return_tensors='pt'):
        return self.encode(text, padding, truncation, max_length, return_tensors)


class MultiModalCollator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.modalities = cfg.MODEL.MULTI_MODAL.MODALITIES

    def __call__(self, batch):
        collated = {}
        for modality in self.modalities:
            data = [item[modality] for item in batch if modality in item and item[modality] is not None]
            collated[modality] = torch.stack(data) if data else None

        collated['text_input_ids'] = torch.stack([item['text_input_ids'] for item in batch])
        collated['text_attention_mask'] = torch.stack([item['text_attention_mask'] for item in batch])

        if 'targets' in batch[0]:
            collated['targets'] = [item['targets'] for item in batch]

        return collated
