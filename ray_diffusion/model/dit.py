# Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py

import math

import ipdb  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_xformers_attention=False,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if use_xformers_attention:
            from ray_diffusion.model.memory_efficient_attention import MEAttention

            attn = MEAttention
        else:
            attn = Attention
        self.attn = attn(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        in_channels=442,
        out_channels=6,
        width=16,
        hidden_size=1152,
        depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        max_num_images=8,
        P=1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.hidden_size = hidden_size
        self.max_num_images = max_num_images
        self.P = P

        self.x_embedder = PatchEmbed(
            img_size=self.width,
            patch_size=self.P,
            in_chans=in_channels,
            embed_dim=hidden_size,
            bias=True,
            flatten=False,
        )
        self.x_pos_enc = FeaturePositionalEncoding(
            max_num_images, hidden_size, width**2, P=self.P
        )
        self.t_embedder = TimestepEmbedder(hidden_size)

        try:
            import xformers

            use_xformers_attention = True
        except ImportError:
            # xformers not available
            use_xformers_attention = False

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    use_xformers_attention=use_xformers_attention,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, P, out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)

        # print("unpatchify", c, p, h, w, x.shape)
        # assert h * w == x.shape[2]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nhpwqc", x)
        imgs = x.reshape(shape=(x.shape[0], h * p, h * p, c))
        return imgs

    def forward(self, x, t):
        """

        Args:
            x: Image/Ray features (B, N, C, H, W).
            t: Timesteps (N,).

        Returns:
            (B, N, D, H, W)
        """
        B, N, c, h, w = x.shape
        P = self.P

        x = x.reshape((B * N, c, h, w))  # (B * N, C, H, W)
        x = self.x_embedder(x)  # (B * N, C, H / P, W / P)

        x = x.permute(0, 2, 3, 1)  # (B * N, H / P, W / P, C)
        # (B, N, H / P, W / P, C)
        x = x.reshape((B, N, h // P, w // P, self.hidden_size))
        x = self.x_pos_enc(x)  # (B, N, H * W / P ** 2, C)
        # TODO: fix positional encoding to work with (N, C, H, W) format.

        # Eval time, we get a scalar t
        if x.shape[0] != t.shape[0] and t.shape[0] == 1:
            t = t.repeat_interleave(B)

        t = self.t_embedder(t)

        for i, block in enumerate(self.blocks):
            x = x.reshape((B, N * h * w // P**2, self.hidden_size))
            x = block(x, t)  # (N, T, D)

        # (B, N * H * W / P ** 2, D)
        x = self.final_layer(
            x, t
        )  # (B, N * H * W / P ** 2,  6 * P ** 2) or (N, T, patch_size ** 2 * out_channels)

        x = x.reshape((B * N, w * w // P**2, self.out_channels * P**2))
        x = self.unpatchify(x)  # (B * N, H, W, C)
        x = x.reshape((B, N) + x.shape[1:])
        x = x.permute(0, 1, 4, 2, 3)  # (B, N, C, H, W)
        return x


class FeaturePositionalEncoding(nn.Module):
    def _get_sinusoid_encoding_table(self, n_position, d_hid, base):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(base, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def __init__(self, max_num_images=8, feature_dim=1152, num_patches=256, P=1):
        super().__init__()
        self.max_num_images = max_num_images
        self.feature_dim = feature_dim
        self.P = P
        self.num_patches = num_patches // self.P**2

        self.register_buffer(
            "image_pos_table",
            self._get_sinusoid_encoding_table(
                self.max_num_images, self.feature_dim, 10000
            ),
        )

        self.register_buffer(
            "token_pos_table",
            self._get_sinusoid_encoding_table(
                self.num_patches, self.feature_dim, 70007
            ),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_images = x.shape[1]

        x = x.reshape(batch_size, num_images, self.num_patches, self.feature_dim)

        # To encode image index
        pe1 = self.image_pos_table[:, :num_images].clone().detach()
        pe1 = pe1.reshape((1, num_images, 1, self.feature_dim))
        pe1 = pe1.repeat((batch_size, 1, self.num_patches, 1))

        # To encode patch index
        pe2 = self.token_pos_table.clone().detach()
        pe2 = pe2.reshape((1, 1, self.num_patches, self.feature_dim))
        pe2 = pe2.repeat((batch_size, num_images, 1, 1))

        x_pe = x + pe1 + pe2
        x_pe = x_pe.reshape(
            (batch_size, num_images * self.num_patches, self.feature_dim)
        )

        return x_pe
