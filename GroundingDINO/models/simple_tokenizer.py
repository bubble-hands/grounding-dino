import torch

class SimpleTokenizer:
    def __init__(self, vocab_size=30522):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.mask_token_id = 3
        self.bos_token_id = 1
        self.eos_token_id = 2
        
        self._char_to_id = {}
        self._id_to_char = {}
        chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?;:\'\"()[]{}<>-=_+*/\\|@#$%^&~`'
        
        idx = 4
        for c in chars:
            self._char_to_id[c] = idx
            self._id_to_char[idx] = c
            idx += 1
        
        self.unk_token_id = idx
        self._char_to_id['[UNK]'] = idx
        self._id_to_char[idx] = '[UNK]'
    
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