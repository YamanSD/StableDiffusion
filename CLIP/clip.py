import torch
from torch import nn
from Attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int) -> None:
        super().__init__()

        self.token_embedding: nn.Embedding = nn.Embedding(n_vocab, n_embd)

        # A learnable weight matrix encodes the position information for each token
        self.position_embedding: nn.Parameter = nn.Parameter(torch.zeros((n_token, n_embd)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x: torch.Tensor = self.token_embedding(tokens)

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x += self.position_embedding

        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()

        # Pre-attention norm
        self.layernorm_1: nn.LayerNorm = nn.LayerNorm(n_embd)

        # Self attention
        self.attention: SelfAttention = SelfAttention(n_head, n_embd)

        # Pre-FNN norm
        self.layernorm_2: nn.LayerNorm = nn.LayerNorm(n_embd)

        # Feedforward layer
        self.linear_1: nn.Linear = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2: nn.Linear = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch_Size, Seq_Len, Dim)
        residue: torch.Tensor = x

        """
            Self Attention
        """

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x: torch.Tensor = self.layernorm_1(x)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x: torch.Tensor = self.attention(x, causal_mask=True)

        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        """
            Feed-Forward Layer
        """

        # Apply a feedforward layer where the hidden dimension is 4 times the embedding dimension.
        residue: torch.Tensor = x

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x: torch.Tensor = self.layernorm_2(x)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x: torch.Tensor = self.linear_1(x)

        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x: torch.Tensor = x * torch.sigmoid(1.702 * x)  # QuickGELU activation function

        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, Dim)
        x: torch.Tensor = self.linear_2(x)

        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        return x


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding: CLIPEmbedding = CLIPEmbedding(
            49408,
            768,
            77
        )

        self.layers: nn.ModuleList = nn.ModuleList([
            CLIPLayer(12, 768) for _ in range(12)
        ])

        self.layernorm: nn.LayerNorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens: torch.Tensor = tokens.type(torch.long)

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state: torch.Tensor = self.embedding(tokens)

        # Apply encoder layers similar to the Transformer's encoder.
        for layer in self.layers:
            # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
            state = layer(state)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output: torch.FloatTensor = self.layernorm(state)

        return output
