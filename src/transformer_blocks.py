import torch
import torch.nn as nn
import math
from torch.nn import MultiheadAttention as MultiHeadAttentionBlock


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class LayerNorm(nn.Module):
    def __init__(self, epsilon: float = 10**-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.gamma * (x - x.mean(dim = -1, keepdim=True)) / (x.std(dim = -1, keepdim=True) + self.epsilon) + self.beta


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        return self.ffn(x)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_head: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.head_dim = d_model // num_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        out = torch.matmul(attn, value)
        return out, attn

    def forward(self, q, k, v, mask):
        batch_size = q.shape[0]
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        query = query.view(batch_size, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(x)



class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, ffn: FeedForwardBlock, p: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.ffn = ffn
        self.norm = LayerNorm()
        self.dropout = nn.Dropout()

    def forward(self, x, src_mask):
        x = x + self.dropout(self.self_attention_block(self.norm(x), self.norm(x), self.norm(x), src_mask))
        x = x + self.dropout(self.ffn(self.norm(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttentionBlock, x_attention: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, p: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.x_attention = x_attention
        self.ffn = feed_forward_block
        self.norm = LayerNorm()
        self.dropout = nn.Dropout(p)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = x + self.dropout(self.self_attention(self.norm(x), self.norm(x), self.norm(x), tgt_mask))
        x = x + self.dropout(self.x_attention(self.norm(x), encoder_output, encoder_output, src_mask))
        x = x + self.dropout(self.ffn(self.norm(x)))
        return x