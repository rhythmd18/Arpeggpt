import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    '''
    Multi-Head Self-Attention Module
    '''
    def __init__(self, d_in, d_out, context_length,
                 dropout, num_heads, qkv_bias=False):
        '''Initialize the Multi-Head Self-Attention module.'''
        super().__init__()
        assert (d_out % num_heads == 0), \
            'd_out must be divisible by num_heads'
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, X, attn_mask=None):
        '''
        Forward pass for Multi-Head Self-Attention.
        Args:
            X: Input tensor of shape (batch_size, num_tokens, d_in)
            attn_mask: Optional attention mask tensor
        Returns:
            context_vec: Output tensor of shape (batch_size, num_tokens, d_out)
        '''
        b, num_tokens, d_in = X.shape
        queries = self.W_query(X)
        keys = self.W_key(X)
        values = self.W_value(X)
        
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        if attn_mask is not None:
            mask = attn_mask[:, None, None, :] == 0
            attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
    

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, X):
        mean = X.mean(dim=-1, keepdim=True)
        var = X.var(dim=-1, keepdims=True, unbiased=False)
        norm_X = (X - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_X + self.shift
    

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh((
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        )))
    

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
        )

    def forward(self, X):
        return self.layers(X)
    

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            dropout=cfg['drop_rate'],
            num_heads=cfg['n_heads'],
            qkv_bias=cfg['qkv_bias']
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, X, attn_mask=None):
        
        shortcut = X
        X = self.norm1(X)
        X = self.attn(X, attn_mask)
        X = self.drop_shortcut(X)
        X = X + shortcut

        shortcut = X
        X = self.norm2(X)
        X = self.ff(X)
        X = self.drop_shortcut(X)
        X = X + shortcut

        return X
    

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_idx, attn_mask=None):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        X = tok_embeds + pos_embeds
        X = self.drop_emb(X)
        for block in self.transformer_blocks:
            X = block(X, attn_mask)
        X = self.final_norm(X)
        X = self.out_head(X)
        return X