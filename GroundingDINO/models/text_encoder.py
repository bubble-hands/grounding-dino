import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_dim=768, vocab_size=30522, max_len=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=12, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        print(f"Created simplified TextEncoder with hidden_dim={hidden_dim}, vocab_size={vocab_size}")

    def forward(self, text_input_ids, text_attention_mask):
        x = self.embedding(text_input_ids)
        x = x + self.positional_encoding[:, :x.size(1), :]
        
        mask = ~text_attention_mask.bool() if text_attention_mask is not None else None
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        return x