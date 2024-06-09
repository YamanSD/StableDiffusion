import torch
from torch import nn
from torch.nn import functional
import math


class SelfAttention(nn.Module):
    def __init__(
            self,
            n_heads: int,
            d_embed: int,
            in_proj_bias: bool = True,
            out_proj_bias: bool = True
    ) -> None:
        """
        :param n_heads: Number of heads.
        :param d_embed: Number of embeddings.
        :param in_proj_bias: In projection bias.
        :param out_proj_bias: Out projection bias.
        """
        super().__init__()

        # This combines the Wq, Wk and Wv matrices into one matrix
        self.in_proj: nn.Linear = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)

        # This one represents the Wo matrix
        self.out_proj: nn.Linear = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads: int = n_heads
        self.d_head: int = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        """
        :param x: (Batch_Size, Seq_Len, Dim)
        :param causal_mask:
        :return:
        """
        # (Batch_Size, Seq_Len, Dim)
        input_shape: tuple[int, ...] = x.shape

        # (Batch_Size, Seq_Len, Dim)
        batch_size, sequence_length, d_embed = input_shape

        # (Batch_Size, Seq_Len, H, Dim / H)
        interim_shape: tuple[int, ...] = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        q: torch.Tensor = q.view(interim_shape).transpose(1, 2)
        k: torch.Tensor = k.view(interim_shape).transpose(1, 2)
        v: torch.Tensor = v.view(interim_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len, Dim / H) @ (Batch_Size, H, Dim / H, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight: torch.Tensor = q @ k.transpose(-1, -2)

        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask: torch.Tensor = torch.ones_like(weight, dtype=torch.bool).triu(1)
            # Fill the upper triangle with -inf
            weight.masked_fill_(mask, -torch.inf)

        # Divide by d_k (Dim / H).
        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight /= math.sqrt(self.d_head)

        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight: torch.Tensor = functional.softmax(weight, dim=-1)

        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output: torch.Tensor = weight @ v

        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output: torch.Tensor = output.transpose(1, 2)

        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output: torch.Tensor = output.reshape(input_shape)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output: torch.Tensor = self.out_proj(output)

        # (Batch_Size, Seq_Len, Dim)
        return output


class CrossAttention(nn.Module):
    def __init__(
            self,
            n_heads: int,
            d_embed: int,
            d_cross: int,
            in_proj_bias: bool = True,
            out_proj_bias: bool = True
    ) -> None:
        super().__init__()
        self.q_proj: nn.Linear = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj: nn.Linear = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj: nn.Linear = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj: nn.Linear = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads: int = n_heads
        self.d_head: int = d_embed // n_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param x: latent of shape (Batch_Size, Seq_Len_Q, Dim_Q)
        :param y: context of shape (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)
        :return:
        """

        input_shape: tuple[int, ...] = x.shape
        batch_size, sequence_length, d_embed = input_shape

        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape: tuple[int, ...] = (batch_size, -1, self.n_heads, self.d_head)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q: torch.Tensor = self.q_proj(x)

        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k: torch.Tensor = self.k_proj(y)

        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v: torch.Tensor = self.v_proj(y)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q,
        # Dim_Q / H)
        q: torch.Tensor = q.view(interim_shape).transpose(1, 2)

        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV,
        # Dim_Q / H)
        k: torch.Tensor = k.view(interim_shape).transpose(1, 2)

        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV,
        # Dim_Q / H)
        v: torch.Tensor = v.view(interim_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H,
        # Seq_Len_Q, Seq_Len_KV)
        weight: torch.Tensor = q @ k.transpose(-1, -2)

        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight /= math.sqrt(self.d_head)

        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = functional.softmax(weight, dim=-1)

        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H,
        # Seq_Len_Q, Dim_Q / H)
        output: torch.Tensor = weight @ v

        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        output: torch.Tensor = output.transpose(1, 2).contiguous()

        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output: torch.Tensor = output.view(input_shape)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output: torch.Tensor = self.out_proj(output)

        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return output
