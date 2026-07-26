import torch
import numpy as np
from PIL import Image
import cv2
from transformers import BertTokenizer

from groundingdino.models.groundingdino import GroundingDINO


class GroundingPredictor:
    def __init__(self, cfg, checkpoint_path=None):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.TEXT_ENCODER.NAME)
        self.max_seq_len = 256

        self.model = GroundingDINO(cfg).to(self.device)

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        self.model.eval()

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Checkpoint loaded from {path}')

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
        if modality == 'depth':
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img.dtype == np.uint16:
                img = (img / 65535.0).astype(np.float32)
            else:
                img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
        else:
            img = Image.open(img_path).convert('RGB' if modality == 'rgb' else 'L')
            img = np.array(img)
            if modality == 'ir':
                img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
            else:
                img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0).to(self.device)

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

        return {
            'pred_logits': outputs['pred_logits'].cpu().numpy(),
            'pred_boxes': outputs['pred_boxes'].cpu().numpy()
        }

    def visualize(self, rgb_path, results, threshold=0.5):
        img = cv2.imread(rgb_path)
        h, w = img.shape[:2]

        boxes = results['pred_boxes'][0]
        logits = results['pred_logits'][0]

        for i in range(boxes.shape[0]):
            box = boxes[i]
            score = logits[i].max()

            if score > threshold:
                x1, y1, x2, y2 = box
                x1 = int(x1 * w)
                y1 = int(y1 * h)
                x2 = int(x2 * w)
                y2 = int(y2 * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{score:.2f}', (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img