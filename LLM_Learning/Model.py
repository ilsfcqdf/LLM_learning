import torch
import torch.nn as nn
import torch.nn.functional as F
import math



# hyperparameters
d_model = 512
num_heads = 8
context_len = 32
batch_size = 4
dropout = 0.1
max_token_value = 200000
max_iter = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'



class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):

        return self.ffn(x)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model//num_heads, bias=False)
        self.Wk = nn.Linear(d_model, d_model//num_heads, bias=False)
        self.Wv = nn.Linear(d_model, d_model//num_heads, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(context_len, context_len)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        attention_score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_model//num_heads)



        attention_score = attention_score.masked_fill(self.mask[:T, :T]==0, float('-inf'))
        attention_score = F.softmax(attention_score, dim=-1)
        # attention_score = self.dropout(dropout)
        attention = attention_score @ v

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Attention() for _ in range(num_heads)])
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.dropout(self.Wo(output))

        return output


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.LN1= nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)
        self.feedforward = FeedForwardNetwork()
        self.MultiHeadAttention = MultiHeadAttention()

    def forward(self, x):
        x = x + self.MultiHeadAttention(self.LN1(x))
        x = x + self.feedforward(self.LN2(x))


        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocal_linear = nn.Linear(d_model, max_token_value)
        self.token_embedding_table = nn.Embedding(max_token_value, d_model)
        self.transformer_blocks = nn.Sequential(
            *([Block() for _ in range(num_heads)] + [nn.LayerNorm(d_model)])
        )

    def forward(self, x_batch, y_batch):
        # print('************************************', x_batch.shape)
        pe_table = torch.zeros(context_len, d_model, device=device)
        position = torch.arange(0, context_len, dtype=torch.long, device=device).unsqueeze(1)

        div_term = torch.exp(
            -torch.log(torch.tensor(10000.0, device=device)) * \
            torch.arange(0, d_model, 2, device=device).float() / d_model)
        # print(div_term.shape)
        pe_table[:, 0::2] = torch.sin(position * div_term)
        pe_table[:, 1::2] = torch.cos(position * div_term)
        # print(pe_table.shape)
        pe_table = pe_table.unsqueeze(0)
        # print('************************************', x_batch.shape)
        output = self.token_embedding_table(x_batch)
        B, T, D = output.shape
        # print('************************************', output.shape)
        output = output+ pe_table
        output = self.transformer_blocks(output)
        logits = self.vocal_linear(output)

        if y_batch is not None:
            B, T, D = logits.shape
            logits_reshaped = logits.view(B * T, D)
            y_batch_reshape = y_batch.view(B * T)
            loss = F.cross_entropy(logits_reshaped, y_batch_reshape)
        else:
            loss = None


        return logits, loss


    def generate(self, x_batch, max_new_tokens=100, temperature=0.7):
        for _ in range(max_new_tokens):
            x_crop = x_batch[:, -context_len:]
            # print(x_crop.shape)
            logits, _ = self.forward(x_crop, None)
            # print(logits.shape)
            logits = logits[:, -1, :] / temperature
            # print(logits.shape)
            probabilities = F.softmax(logits, dim=-1)
            # print('********', probabilities)
            predicted_token = torch.multinomial(probabilities, num_samples=1)

            x_batch = torch.cat([x_batch, predicted_token], dim=1)


        return x_batch










