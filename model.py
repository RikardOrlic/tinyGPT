import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embedding % config.n_head == 0
        self.head_size = config.n_embedding // config.n_head

        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size))) # (T, T)

        self.query = nn.Linear(config.n_embedding, self.head_size, bias=False)
        self.key = nn.Linear(config.n_embedding, self.head_size, bias=False)
        self.value = nn.Linear(config.n_embedding, self.head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape # x = (B, T, C)
        q = self.query(x) # (B, T, H)
        k = self.key(x)   # (B, T, H)

        att = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)     # (B, T, H) @ (B, H, T) -> (B, T, T)
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        att = F.softmax(att, dim=-1)                                 # (B, T, T)
        att = self.dropout(att)

        v = self.value(x) # (B, T, H)

        out = att @ v     # (B, T, T) @ (B, T, H) -> (B, T, H)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([MaskAttentionHead(config) for _ in range(config.n_head)])
        self.fc = nn.Linear(config.n_embedding, config.n_embedding)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = [h(x) for h in self.heads] # [(B, T, H)] * n_head
        x = torch.cat(x, dim=-1)       # (B, T, H * n_head)

        x = self.fc(x)                 # (B, T, C)
        out = self.dropout(x)
        return out


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embedding, config.n_embedding * 4, bias = True)
        self.relu = nn.GELU()
        self.fc2 = nn.Linear(config.n_embedding * 4, config.n_embedding, bias = True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        out = self.dropout(x)
    
        return out


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn = MultiHeadAttention(config)
        self.lnorm1 = nn.LayerNorm(config.n_embedding, bias=False) 
        self.ff = FeedForward(config)
        self.lnorm2 = nn.LayerNorm(config.n_embedding, bias=False) 

    def forward(self, x):
        x = x + self.attn(self.lnorm1(x)) #Add & Norm
        out = x + self.ff(self.lnorm2(x)) #Add & Norm

        return out
    
        
class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_embd = nn.Embedding(config.vocab_size, config.n_embedding)
        self.pos_embd = nn.Embedding(config.block_size, config.n_embedding)

        self.layers = nn.Sequential(*[DecoderBlock(config) for _ in range(config.n_blocks)])

        self.lnorm = nn.LayerNorm(config.n_embedding)
        self.fc = nn.Linear(config.n_embedding, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        #small weights so the model is less confidently incorrect about the predictions on the start
        #random character selection loss = -ln(1/vocab_size)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        x = self.tok_embd(x) + self.pos_embd(torch.arange(x.size(1), device=self.config.device)) # (B, T, C)
        x = self.layers(x)
        x = self.lnorm(x)
        out = self.fc(x)

        return out
    
    @torch.no_grad
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    