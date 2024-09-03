import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(AdditiveAttention, self).__init__()
        self.query_linear = nn.Linear(in_size, hidden_size)
        self.key_linear = nn.Linear(in_size, hidden_size)
        self.energy_linear = nn.Linear(hidden_size, 1)
        self.final_linear = nn.Linear(in_size, in_size)

    def forward(self, z):
        query = self.query_linear(z)
        key = self.key_linear(z)
        energy = self.energy_linear(torch.tanh(query + key))
        attention_weights = F.softmax(energy, dim=1)
        output = torch.matmul(attention_weights.transpose(1, 2), z)
        output = output.mean(dim=1)
        output = self.final_linear(output)
        return output, attention_weights


class GatedAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(GatedAttention, self).__init__()

        self.query_linear = nn.Linear(in_size, hidden_size)
        self.key_linear = nn.Linear(in_size, hidden_size)
        self.value_linear = nn.Linear(in_size, hidden_size)
        self.gate_linear = nn.Linear(in_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.final_linear = nn.Linear(hidden_size, in_size)

    def forward(self, z):
        query = self.query_linear(z)
        key = self.key_linear(z)
        value = self.value_linear(z)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        gate = self.sigmoid(self.gate_linear(z))
        gated_value = value * gate
        gated_attention_output = torch.matmul(attention_weights, gated_value)
        output = self.final_linear(gated_attention_output.sum(dim=1))
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, in_size, num_heads=4, hidden_size=256):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.query_linear = nn.Linear(in_size, hidden_size)
        self.key_linear = nn.Linear(in_size, hidden_size)
        self.value_linear = nn.Linear(in_size, hidden_size)

        self.final_linear = nn.Linear(hidden_size, in_size)

    def forward(self, z):
        query = self.query_linear(z)
        key = self.key_linear(z)
        value = self.value_linear(z)

        query = query.view(query.size(0), -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(key.size(0), -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(value.size(0), -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)

        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(z.size(0), -1, self.hidden_size)
        output = attention_output.mean(dim=1)
        output = self.final_linear(output)
        return output, attention_weights


class PositionAwareAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128, max_seq_length=200):
        super(PositionAwareAttention, self).__init__()

        self.query_linear = nn.Linear(in_size, hidden_size)
        self.key_linear = nn.Linear(in_size, hidden_size)
        self.value_linear = nn.Linear(in_size, hidden_size)

        self.position_encoding = nn.Embedding(max_seq_length, hidden_size)

        self.final_linear = nn.Linear(hidden_size, in_size)

    def forward(self, z):
        query = self.query_linear(z)
        key = self.key_linear(z)
        value = self.value_linear(z)

        seq_length = z.size(1)
        positions = torch.arange(seq_length, device=z.device)
        position_encoding = self.position_encoding(positions)
        query += position_encoding.unsqueeze(0)
        key += position_encoding.unsqueeze(0)

        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(query.size(-1), dtype=torch.float32))
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)
        output = attention_output.mean(dim=1)
        output = self.final_linear(output)

        return output, attention_weights


class ScaledDotProductAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(ScaledDotProductAttention, self).__init__()
        self.query_linear = nn.Linear(in_size, hidden_size)
        self.key_linear = nn.Linear(in_size, hidden_size)
        self.value_linear = nn.Linear(in_size, hidden_size)
        self.final_linear = nn.Linear(hidden_size, in_size)


    def forward(self, z):
        query = self.query_linear(z)
        key = self.key_linear(z)
        value = self.value_linear(z)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, value)
        output = output.mean(dim=1)
        output = self.final_linear(output)
        return output, attention_weights


class SelfAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SelfAttention, self).__init__()
        self.query_linear = nn.Linear(in_size, hidden_size)
        self.key_linear = nn.Linear(in_size, hidden_size)
        self.value_linear = nn.Linear(in_size, hidden_size)
        self.final_linear = nn.Linear(hidden_size, in_size)
    def forward(self, z):
        # Linear transformations
        query = self.query_linear(z)
        key = self.key_linear(z)
        value = self.value_linear(z)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)
        output = attention_output.mean(dim=1)
        output = self.final_linear(output)
        return output, attention_weights