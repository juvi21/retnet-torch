class ModelArgs:
    
    def __init__(self, dim=24, vocab_size=8, num_layers=6, num_heads=6, dim_ffn=None, 
                 norm_eps=1e-6, dropout=0.0, layer_args=None):
        self.dim = dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_ffn = dim_ffn
        self.norm_eps = norm_eps
        self.dropout = dropout
        self.layer_args = layer_args or {}