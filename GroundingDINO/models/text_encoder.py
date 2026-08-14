import os
import torch
import torch.nn as nn

# 设置 HuggingFace 镜像，解决网络超时问题
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

try:
    from transformers import BertModel, BertConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_dim=768, vocab_size=30522, max_len=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.bert = None
        self._is_pretrained = False

        if HAS_TRANSFORMERS:
            bert_loaded = False
            # 1. 先尝试本地缓存（已下载完整权重时）
            try:
                self.bert = BertModel.from_pretrained(model_name, local_files_only=True)
                print(f"[TextEncoder] Loaded BERT from local cache: {model_name}")
                bert_loaded = True
                self._is_pretrained = True
            except Exception as e:
                print(f"[TextEncoder] Local cache miss ({e}), trying mirror...")

            # 2. 本地缓存失败，通过镜像下载预训练权重
            if not bert_loaded:
                try:
                    self.bert = BertModel.from_pretrained(model_name)
                    print(f"[TextEncoder] Loaded BERT from mirror: {model_name}")
                    bert_loaded = True
                    self._is_pretrained = True
                except Exception as e2:
                    print(f"[TextEncoder] Mirror also failed ({e2}), using random init")

            # 3. 全部失败，随机初始化
            if not bert_loaded:
                config = BertConfig(
                    vocab_size=vocab_size,
                    hidden_size=hidden_dim,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    max_position_embeddings=512,
                    type_vocab_size=2,
                )
                self.bert = BertModel(config)
                print(f"[TextEncoder] Created BERT with random weights: hidden_dim={hidden_dim}")
        else:
            print("[TextEncoder] transformers not available, using fallback embedding")
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self.pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_dim))
            layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=12, batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=6)

        if not self._is_pretrained:
            self._init_weights()

    def _init_weights(self):
        if self.bert is not None:
            for p in self.bert.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, text_input_ids, text_attention_mask):
        if self.bert is not None:
            outputs = self.bert(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
                token_type_ids=None,
            )
            return outputs.last_hidden_state
        else:
            x = self.embedding(text_input_ids)
            x = x + self.pos_embed[:, :x.size(1), :]
            mask = ~text_attention_mask.bool() if text_attention_mask is not None else None
            x = self.transformer(x, src_key_padding_mask=mask)
            return x
