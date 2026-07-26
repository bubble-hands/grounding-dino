import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2


class SimpleTokenizer:
    def __init__(self, vocab_size=30522):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.mask_token_id = 3
        
        self._char_to_id = {}
        self._id_to_char = {}
        chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?;:\'\"()[]{}<>-=_+*/\\|@#$%^&~`'
        
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


class MultiModalDataset(Dataset):
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split
        self.data_path = cfg.DATASETS.DATA_PATH
        self.modalities = cfg.MODEL.MULTI_MODAL.MODALITIES

        self.tokenizer = SimpleTokenizer()
        self.max_seq_len = 256

        self.data = self._load_data()

    def _load_data(self):
        ann_file = os.path.join(self.data_path, f'{self.split}.json')
        with open(ann_file, 'r') as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        inputs = {}
        for modality in self.modalities:
            if modality in item and item[modality] is not None:
                img_path = os.path.join(self.data_path, item[modality])
                inputs[modality] = self._load_image(img_path, modality)

        text = item.get('text', item.get('prompt', ''))
        text_input_ids, text_attention_mask = self._tokenize(text)
        inputs['text_input_ids'] = text_input_ids
        inputs['text_attention_mask'] = text_attention_mask

        if self.split == 'train' and 'annotations' in item:
            targets = self._prepare_targets(item['annotations'])
            inputs['targets'] = targets

        return inputs

    def _load_image(self, img_path, modality):
        target_size = (512, 512)
        
        if modality == 'depth':
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
            if img.dtype == np.uint16:
                img = (img / 65535.0).astype(np.float32)
            else:
                img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
        else:
            img = Image.open(img_path).convert('RGB' if modality == 'rgb' else 'L')
            img = img.resize(target_size, Image.BILINEAR)
            img = np.array(img)
            if modality == 'ir':
                img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
            else:
                img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.from_numpy(img)

    def _tokenize(self, text):
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors='pt'
        )
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)

    def _prepare_targets(self, annotations):
        labels = []
        boxes = []
        for ann in annotations:
            labels.append(ann['category_id'])
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
        return {'labels': torch.tensor(labels, dtype=torch.long),
                'boxes': torch.tensor(boxes, dtype=torch.float32)}


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