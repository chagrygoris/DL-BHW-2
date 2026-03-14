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
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        out = self.linear2(x)
        return out


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_head: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head

        self.d_k = d_model // num_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)
        attention_score = attention_score.softmax(dim=-1)
        if dropout is not None:
            attention_score = dropout(attention_score)

        return (attention_score @ value), attention_score

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        query = query.view(query.shape[0], query.shape[1], self.num_head, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.num_head, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.num_head, self.d_k).transpose(1,2)

        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.num_head*self.d_k)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.norm = LayerNorm()
        self.dropout = nn.Dropout(p)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.projection(x), dim=-1)


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, ffn: FeedForwardBlock, p: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.ffn = ffn
        self.residual_connection = nn.ModuleList([ResidualConnection(p) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.ffn)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttentionBlock, x_attention: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, p: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.x_attention = x_attention
        self.ffn = feed_forward_block
        self.norm = LayerNorm()
        self.dropout = nn.Dropout(p)
        self.residual_connection = nn.ModuleList([ResidualConnection(p) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = x + self.dropout(self.self_attention(self.norm(x), self.norm(x), self.norm(x), tgt_mask))
        # x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = x + self.dropout(self.x_attention(self.norm(x), encoder_output, encoder_output, src_mask))
        # x = self.residual_connection[1](x, lambda x: self.x_attention(x, encoder_output, encoder_output, src_mask))
        x = x + self.dropout(self.ffn(self.norm(x)))
        # x = self.residual_connection[2](x, self.ffn)
        return x