import os
import torch
import numpy as np
from PIL import Image
import cv2

try:
    from transformers import BertTokenizerFast
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from groundingdino.models.groundingdino import GroundingDINO


class GroundingPredictor:
    def __init__(self, cfg, checkpoint_path=None):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = self._create_tokenizer(cfg.MODEL.TEXT_ENCODER.NAME)
        self.max_seq_len = cfg.MODEL.get('MAX_SEQ_LEN', 256)
        self.target_size = (512, 512)

        self.model = GroundingDINO(cfg).to(self.device)

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)

        self.model.eval()
        print(f"[Predictor] Model loaded on {self.device}")

    def _create_tokenizer(self, model_name):
        if HAS_TRANSFORMERS:
            try:
                tokenizer = BertTokenizerFast.from_pretrained(model_name, local_files_only=True)
                print(f"[Predictor] Loaded BertTokenizerFast from local cache: {model_name}")
                return tokenizer
            except Exception:
                pass

            try:
                tokenizer = BertTokenizerFast.from_pretrained(model_name)
                print(f"[Predictor] Loaded BertTokenizerFast: {model_name}")
                return tokenizer
            except Exception as e:
                print(f"[Predictor] Cannot load tokenizer ({e}), using fallback")

        from groundingdino.datasets.dataset import SimpleTokenizer
        print("[Predictor] Using SimpleTokenizer fallback")
        return SimpleTokenizer()

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f'[Predictor] Checkpoint loaded from {path}')

    def _tokenize(self, text):
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors='pt'
        )
        return encoding['input_ids'], encoding['attention_mask']

    def _prepare_image(self, img_path, modality):
        if not os.path.exists(img_path):
            return None

        target_h, target_w = self.target_size

        if modality == 'depth':
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_NEAREST)
            if img.dtype == np.uint16:
                img = (img / 65535.0).astype(np.float32)
            else:
                img = img.astype(np.float32) / 255.0
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            else:
                img = img.transpose(2, 0, 1)
        elif modality == 'ir':
            img = Image.open(img_path).convert('L')
            img = img.resize(self.target_size, Image.BILINEAR)
            img = np.array(img).astype(np.float32) / 255.0
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
        else:
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            img = img.resize(self.target_size, Image.BILINEAR)
            img = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0

        return torch.from_numpy(img.copy()).unsqueeze(0).to(self.device)

    def predict(self, rgb_path=None, ir_path=None, depth_path=None, text_prompt=None):
        inputs = {}

        if rgb_path is not None:
            inputs['rgb'] = self._prepare_image(rgb_path, 'rgb')
        if ir_path is not None:
            inputs['ir'] = self._prepare_image(ir_path, 'ir')
        if depth_path is not None:
            inputs['depth'] = self._prepare_image(depth_path, 'depth')

        if text_prompt is not None:
            text_input_ids, text_attention_mask = self._tokenize(text_prompt)
            inputs['text_input_ids'] = text_input_ids.to(self.device)
            inputs['text_attention_mask'] = text_attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)

        return outputs

    def visualize(self, rgb_path, results, threshold=0.5):
        img = cv2.imread(rgb_path)
        if img is None:
            return None
        h, w = img.shape[:2]

        if isinstance(results, dict) and 'pred_boxes' in results:
            boxes = results['pred_boxes'][0]
            logits = results['pred_logits'][0]

            for i in range(boxes.shape[0]):
                box = boxes[i]
                score = float(torch.sigmoid(torch.tensor(logits[i])).max())

                if score > threshold:
                    x1, y1, x2, y2 = box
                    x1 = int(x1 * w)
                    y1 = int(y1 * h)
                    x2 = int(x2 * w)
                    y2 = int(y2 * h)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f'{score:.2f}', (x1, max(y1 - 10, 0)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img
