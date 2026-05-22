import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device = {device}")

B = 128
T = 10
d_k = 16
d_v = 16

Q = torch.randn(B, T, d_k)
K = torch.randn(B, T, d_k)
V = torch.randn(B, T, d_v)

output = F.scaled_dot_product_attention(Q, K, V)
print(f"\nScaled dot-product attention :")
print(f"  output shape : {output.shape}")  



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        i = torch.arange(0, d_model, 2)
        div_term = torch.exp(i * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        T = x.size(1)
        x = x + self.pe[:, :T]
        return self.dropout(x)


d_model = 256
T = 10
B = 128
pe = PositionalEncoding(d_model, dropout=0.0)
x = torch.zeros(B, T, d_model)
out = pe(x)
print(f"\nPositional encoding :")
print(f"  output shape : {out.shape}")   

num_heads = 8
mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, bias=False, batch_first=True)

x = torch.randn(B, T, d_model)
out, attn = mha(x, x, x)   # self-attention : x_q = x_k = x_v = x
print(f"\nMulti-head attention :")
print(f"  output shape : {out.shape}")   # (128, 10, 256)
print(f"  attn shape   : {attn.shape}")  # (128, 10, 10) averaged over heads by default



d_ff = 4 * d_model
block = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=num_heads,
    dim_feedforward=d_ff,
    dropout=0.1,
    activation='relu',
    batch_first=True,
)
y = block(x)
print(f"\nTransformer block :")
print(f"  output shape : {y.shape}")   # (128, 10, 256)


num_layers = 6
encoder = nn.TransformerEncoder(block, num_layers=num_layers)
y = encoder(x)
print(f"\nTransformer encoder ({num_layers} layers) :")
print(f"  output shape : {y.shape}")   # (128, 10, 256)
print(f"  n_params     : {sum(p.numel() for p in encoder.parameters()):,}")
