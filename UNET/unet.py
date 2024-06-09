import torch
from torch import nn
from torch.nn import functional
from Attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.linear_1: nn.Linear = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2: nn.Linear = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (1, 320)
        :return:
        """

        # (1, 320) -> (1, 1280)
        x: torch.Tensor = self.linear_1(x)

        # (1, 1280) -> (1, 1280)
        x: torch.Tensor = functional.silu(x)

        # (1, 1280) -> (1, 1280)
        x: torch.Tensor = self.linear_2(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_time: int = 1280
    ) -> None:
        super().__init__()
        self.groupnorm_feature: nn.GroupNorm = nn.GroupNorm(32, in_channels)
        self.conv_feature: nn.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time: nn.Linear = nn.Linear(n_time, out_channels)

        self.groupnorm_merged: nn.GroupNorm = nn.GroupNorm(32, out_channels)
        self.conv_merged: nn.Conv2d = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer: nn.Identity = nn.Identity()
        else:
            self.residual_layer: nn.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        :param feature: (Batch_Size, In_Channels, Height, Width)
        :param time: (1, 1280)
        :return:
        """

        residue: torch.Tensor = feature

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature: torch.Tensor = self.groupnorm_feature(feature)

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature: torch.Tensor = functional.silu(feature)

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        feature: torch.Tensor = self.conv_feature(feature)

        # (1, 1280) -> (1, 1280)
        time: torch.Tensor = functional.silu(time)

        # (1, 1280) -> (1, Out_Channels)
        time: torch.Tensor = self.linear_time(time)

        # Add width and height dimension to time. (Batch_Size, Out_Channels, Height, Width) + (1, Out_Channels, 1,
        # 1) -> (Batch_Size, Out_Channels, Height, Width)
        merged: torch.Tensor = feature + time.unsqueeze(-1).unsqueeze(-1)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged: torch.Tensor = self.groupnorm_merged(merged)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged: torch.Tensor = functional.silu(merged)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.conv_merged(merged)

        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size,
        # Out_Channels, Height, Width)
        return merged + self.residual_layer(residue)


class AttentionBlock(nn.Module):
    def __init__(
            self,
            n_head: int,
            n_embd: int,
            d_context: int = 768
    ) -> None:
        super().__init__()
        channels: int = n_head * n_embd

        self.groupnorm: nn.GroupNorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input: nn.Conv2d = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1: nn.LayerNorm = nn.LayerNorm(channels)
        self.attention_1: SelfAttention = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2: nn.LayerNorm = nn.LayerNorm(channels)
        self.attention_2: CrossAttention = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3: nn.LayerNorm = nn.LayerNorm(channels)
        self.linear_geglu_1: nn.Linear = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2: nn.Linear = nn.Linear(4 * channels, channels)

        self.conv_output: nn.Conv2d = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        :param x: (Batch_Size, Features, Height, Width)
        :param context: (Batch_Size, Seq_Len, Dim)
        :return:
        """

        residue_long: torch.Tensor = x

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x: torch.Tensor = self.groupnorm(x)

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x: torch.Tensor = self.conv_input(x)

        n, c, h, w = x.shape

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x: torch.Tensor = x.view((n, c, h * w))

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x: torch.Tensor = x.transpose(-1, -2)

        # Normalization + Self-Attention with skip connection

        # (Batch_Size, Height * Width, Features)
        residue_short: torch.Tensor = x

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x: torch.Tensor = self.layernorm_1(x)

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x: torch.Tensor = self.attention_1(x)

        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size,
        # Height * Width, Features)
        x += residue_short

        # (Batch_Size, Height * Width, Features)
        residue_short: torch.Tensor = x

        # Normalization + Cross-Attention with skip connection

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x: torch.Tensor = self.layernorm_2(x)

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x: torch.Tensor = self.attention_2(x, context)

        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size,
        # Height * Width, Features)
        x += residue_short

        # (Batch_Size, Height * Width, Features)
        residue_short: torch.Tensor = x

        # Normalization + FFN with GeGLU and skip connection

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x: torch.Tensor = self.layernorm_3(x)

        # GeGLU as implemented in the original code:
        # https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules
        # /attention.py#L37C10-L37C10 (Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size,
        # Height * Width, Features * 4)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)

        # Element-wise product: (Batch_Size, Height * Width, Features * 4) * (Batch_Size, Height * Width, Features *
        # 4) -> (Batch_Size, Height * Width, Features * 4)
        x: torch.Tensor = x * functional.gelu(gate)

        # (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features)
        x: torch.Tensor = self.linear_geglu_2(x)

        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size,
        # Height * Width, Features)
        x += residue_short

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x: torch.Tensor = x.transpose(-1, -2)

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x: torch.Tensor = x.view((n, c, h, w))

        # Final skip connection between initial input and output of the block (Batch_Size, Features, Height,
        # Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        return self.conv_output(x) + residue_long


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensoer:
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        x: torch.Tensor = functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    def forward(
            self,
            x: torch.Tensor,
            context: torch.Tensor,
            time: torch.Tensor
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, AttentionBlock):
                x: torch.Tensor = layer(x, context)
            elif isinstance(layer, ResidualBlock):
                x: torch.Tensor = layer(x, time)
            else:
                x: torch.Tensor = layer(x)
        return x


class UNET(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoders: nn.ModuleList = nn.ModuleList([
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size,
            # 320, Height / 8, Width / 8)
            SwitchSequential(
                ResidualBlock(320, 320),
                AttentionBlock(8, 40)
            ),

            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size,
            # 320, Height / 8, Width / 8)
            SwitchSequential(
                ResidualBlock(320, 320),
                AttentionBlock(8, 40)
            ),

            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            # (Batch_Size, 320, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (
            # Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(
                ResidualBlock(320, 640),
                AttentionBlock(8, 80)
            ),

            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (
            # Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(
                ResidualBlock(640, 640),
                AttentionBlock(8, 80)
            ),

            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            # (Batch_Size, 640, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (
            # Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(
                ResidualBlock(640, 1280),
                AttentionBlock(8, 160)
            ),

            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (
            # Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(
                ResidualBlock(1280, 1280),
                AttentionBlock(8, 160)
            ),

            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(ResidualBlock(1280, 1280)),

            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(ResidualBlock(1280, 1280)),
        ])

        self.bottleneck: SwitchSequential = SwitchSequential(
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            ResidualBlock(1280, 1280),

            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            AttentionBlock(8, 160),

            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            ResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(ResidualBlock(2560, 1280)),

            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(ResidualBlock(2560, 1280)),

            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (
            # Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(
                ResidualBlock(2560, 1280),
                Upsample(1280)
            ),

            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (
            # Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(
                ResidualBlock(2560, 1280),
                AttentionBlock(8, 160)
            ),

            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (
            # Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(
                ResidualBlock(2560, 1280),
                AttentionBlock(8, 160)
            ),

            # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (
            # Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
            SwitchSequential(
                ResidualBlock(1920, 1280),
                AttentionBlock(8, 160), Upsample(1280)
            ),

            # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (
            # Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(
                ResidualBlock(1920, 640),
                AttentionBlock(8, 80)
            ),

            # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (
            # Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(
                ResidualBlock(1280, 640),
                AttentionBlock(8, 80)
            ),

            # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (
            # Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
            SwitchSequential(
                ResidualBlock(960, 640),
                AttentionBlock(8, 80), Upsample(640)
            ),

            # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size,
            # 320, Height / 8, Width / 8)
            SwitchSequential(
                ResidualBlock(960, 320),
                AttentionBlock(8, 40)
            ),

            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size,
            # 320, Height / 8, Width / 8)
            SwitchSequential(
                ResidualBlock(640, 320),
                AttentionBlock(8, 40)
            ),

            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size,
            # 320, Height / 8, Width / 8)
            SwitchSequential(
                ResidualBlock(640, 320),
                AttentionBlock(8, 40)
            ),
        ])

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        :param x: (Batch_Size, 4, Height / 8, Width / 8)
        :param context: (Batch_Size, Seq_Len, Dim)
        :param time: (1, 1280)
        :return:
        """

        skip_connections: list[torch.Tensor] = []
        for layers in self.encoders:
            x: torch.Tensor = layers(x, context, time)
            skip_connections.append(x)

        x: torch.Tensor = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before
            # being sent to the decoder's layer
            x: torch.Tensor = torch.cat((x, skip_connections.pop()), dim=1)
            x: torch.Tensor = layers(x, context, time)

        return x


class OutputLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ) -> None:
        super().__init__()
        self.groupnorm: nn.GroupNorm = nn.GroupNorm(32, in_channels)
        self.conv: nn.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (Batch_Size, 320, Height / 8, Width / 8)
        :return:
        """
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x: torch.Tensor = self.groupnorm(x)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x: torch.Tensor = functional.silu(x)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x: torch.Tensor = self.conv(x)

        # (Batch_Size, 4, Height / 8, Width / 8)
        return x


class Diffusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.time_embedding: TimeEmbedding = TimeEmbedding(320)
        self.unet: UNET = UNET()
        self.final: OutputLayer = OutputLayer(320, 4)

    def forward(
            self,
            latent: torch.Tensor,
            context: torch.Tensor,
            time: torch.Tensor
    ) -> torch.Tensor:
        """
        :param latent: (Batch_Size, 4, Height / 8, Width / 8)
        :param context: (Batch_Size, Seq_Len, Dim)
        :param time: (1, 320)
        :return:
        """

        # (1, 320) -> (1, 1280)
        time: torch.Tensor = self.time_embedding(time)

        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        output: torch.Tensor = self.unet(latent, context, time)

        # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        output: torch.Tensor = self.final(output)

        # (Batch, 4, Height / 8, Width / 8)
        return output
