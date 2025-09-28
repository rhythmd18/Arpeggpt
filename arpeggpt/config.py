GPT_SMALL_CONFIG = {
    'vocab_size': 30000,    # Vocabulary size
    'context_length': 1024, # Context length
    'emb_dim': 768,         # Embedding dimension
    'n_heads': 12,          # Number of attention heads
    'n_layers': 12,         # Number of layers
    'drop_rate': 0.1,       # Dropout rate
    'qkv_bias': False       # Query-Key-value bias
}

GPT_MEDIUM_CONFIG = {
    'vocab_size': 30000,    # Vocabulary size
    'context_length': 1024, # Context length
    'emb_dim': 1024,         # Embedding dimension
    'n_heads': 16,          # Number of attention heads
    'n_layers': 24,         # Number of layers
    'drop_rate': 0.1,       # Dropout rate
    'qkv_bias': False       # Query-Key-value bias
}

GPT_LARGE_CONFIG = {
    'vocab_size': 30000,    # Vocabulary size
    'context_length': 1024, # Context length
    'emb_dim': 1260,         # Embedding dimension
    'n_heads': 20,          # Number of attention heads
    'n_layers': 36,         # Number of layers
    'drop_rate': 0.1,       # Dropout rate
    'qkv_bias': False       # Query-Key-value bias
}

GPT_XL_CONFIG = {
    'vocab_size': 30000,    # Vocabulary size
    'context_length': 1024, # Context length
    'emb_dim': 1600,         # Embedding dimension
    'n_heads': 25,          # Number of attention heads
    'n_layers': 48,         # Number of layers
    'drop_rate': 0.1,       # Dropout rate
    'qkv_bias': False       # Query-Key-value bias
}

def get_config(size: str):
    if size.lower() == 'small':
        return GPT_SMALL_CONFIG
    elif size.lower() == 'medium':
        return GPT_MEDIUM_CONFIG
    elif size.lower() == 'large':
        return GPT_LARGE_CONFIG
    else:
        return GPT_XL_CONFIG