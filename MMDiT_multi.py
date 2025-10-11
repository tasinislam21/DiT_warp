from __future__ import annotations

from torch import nn
from torch import Tensor
from torch.nn import Module, ModuleList
from timm.models.vision_transformer import PatchEmbed
import torch
from einops.layers.torch import Rearrange
from models import TimestepEmbedder, FinalLayer

from x_transformers import (
    RMSNorm,
    FeedForward
)

from MMDiT import JointAttention, get_2d_sincos_pos_embed

from hyper_connections import (
    HyperConnections,
    Residual
)


# helpers

def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# adaptive layernorm
# aim for clarity in generalized version

class AdaptiveLayerNorm(Module):
    def __init__(
            self,
            dim,
            dim_cond=None
    ):
        super().__init__()
        has_cond = exists(dim_cond)
        self.has_cond = has_cond

        self.ln = nn.LayerNorm(dim, elementwise_affine=not has_cond)

        if has_cond:
            cond_linear = nn.Linear(dim_cond, dim * 2)

            self.to_cond = nn.Sequential(
                Rearrange('b d -> b 1 d'),
                nn.SiLU(),
                cond_linear
            )

            nn.init.zeros_(cond_linear.weight)

            nn.init.constant_(cond_linear.bias[:dim], 1.)
            nn.init.zeros_(cond_linear.bias[dim:])

    def forward(
            self,
            x,
            cond=None
    ):
        assert not (exists(
            cond) ^ self.has_cond), 'condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

        x = self.ln(x)

        if self.has_cond:
            gamma, beta = self.to_cond(cond).chunk(2, dim=-1)
            x = x * gamma + beta

        return x


# class

class MMDiTBlock(Module):
    def __init__(
            self,
            *,
            dim_modalities: tuple[int, ...],
            dim_cond=None,
            dim_head=64,
            heads=8,
            qk_rmsnorm=False,
            flash_attn=False,
            softclamp=False,
            softclamp_value=50.,
            num_residual_streams=1,
            ff_kwargs: dict = dict()
    ):
        super().__init__()
        self.num_modalities = len(dim_modalities)
        self.dim_modalities = dim_modalities

        # residuals / maybe hyper connections

        residual_klass = Residual if num_residual_streams == 1 else HyperConnections

        self.attn_residual_fns = ModuleList([residual_klass(num_residual_streams, dim=dim) for dim in dim_modalities])
        self.ff_residual_fns = ModuleList([residual_klass(num_residual_streams, dim=dim) for dim in dim_modalities])

        # handle optional time conditioning

        has_cond = exists(dim_cond)
        self.has_cond = has_cond

        if has_cond:
            cond_linear = nn.Linear(dim_cond, sum(dim_modalities) * 2)

            self.to_post_branch_gammas = nn.Sequential(
                Rearrange('b d -> b 1 d'),
                nn.SiLU(),
                cond_linear
            )

            nn.init.zeros_(cond_linear.weight)
            nn.init.constant_(cond_linear.bias, 1.)

        # joint modality attention

        attention_layernorms = [AdaptiveLayerNorm(dim, dim_cond=dim_cond) for dim in dim_modalities]
        self.attn_layernorms = ModuleList(attention_layernorms)

        self.joint_attn = JointAttention(
            dim_inputs=dim_modalities,
            dim_head=dim_head,
            heads=heads,
            flash=flash_attn,
            softclamp=softclamp,
            softclamp_value=softclamp_value,
        )

        # feedforwards

        feedforward_layernorms = [AdaptiveLayerNorm(dim, dim_cond=dim_cond) for dim in dim_modalities]
        self.ff_layernorms = ModuleList(feedforward_layernorms)

        feedforwards = [FeedForward(dim, **ff_kwargs) for dim in dim_modalities]
        self.feedforwards = ModuleList(feedforwards)

    def forward(
            self,
            *,
            modality_tokens: tuple[Tensor, ...],
            modality_masks: tuple[Tensor | None, ...] | None = None,
            time_cond=None
    ):
        assert len(modality_tokens) == self.num_modalities
        assert not (exists(
            time_cond) ^ self.has_cond), 'condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

        ln_kwargs = dict()

        if self.has_cond:
            ln_kwargs = dict(cond=time_cond)

            gammas = self.to_post_branch_gammas(time_cond)
            attn_gammas, ff_gammas = gammas.chunk(2, dim=-1)

        # attention layernorms

        modality_tokens, modality_tokens_residual_fns = tuple(
            zip(*[residual_fn(modality_token) for residual_fn, modality_token in
                  zip(self.attn_residual_fns, modality_tokens)]))

        modality_tokens = [ln(tokens, **ln_kwargs) for tokens, ln in zip(modality_tokens, self.attn_layernorms)]

        # attention

        modality_tokens = self.joint_attn(inputs=modality_tokens, masks=modality_masks)

        # post attention gammas

        if self.has_cond:
            attn_gammas = attn_gammas.split(self.dim_modalities, dim=-1)
            modality_tokens = [(tokens * g) for tokens, g in zip(modality_tokens, attn_gammas)]

        # add attention residual

        modality_tokens = [add_attn_residual(tokens) for add_attn_residual, tokens in
                           zip(modality_tokens_residual_fns, modality_tokens)]

        # handle feedforward adaptive layernorm

        modality_tokens, modality_tokens_residual_fns = tuple(
            zip(*[residual_fn(modality_token) for residual_fn, modality_token in
                  zip(self.ff_residual_fns, modality_tokens)]))

        modality_tokens = [ln(tokens, **ln_kwargs) for tokens, ln in zip(modality_tokens, self.ff_layernorms)]

        modality_tokens = [ff(tokens) for tokens, ff in zip(modality_tokens, self.feedforwards)]

        # post feedforward gammas

        if self.has_cond:
            ff_gammas = ff_gammas.split(self.dim_modalities, dim=-1)
            modality_tokens = [(tokens * g) for tokens, g in zip(modality_tokens, ff_gammas)]

        # add feedforward residual

        modality_tokens = [add_residual_fn(tokens) for add_residual_fn, tokens in
                           zip(modality_tokens_residual_fns, modality_tokens)]

        # returns

        return modality_tokens


# mm dit transformer - simply many blocks

class MMDiT(Module):
    def __init__(
            self,
            *,
            depth,
            dim_modalities,
            final_norms=True,
            num_residual_streams=4,
            **block_kwargs
    ):
        super().__init__()

        self.expand_streams, self.reduce_streams = HyperConnections.get_expand_reduce_stream_functions(
            num_residual_streams, disable=num_residual_streams == 1)

        blocks = [MMDiTBlock(dim_modalities=dim_modalities, num_residual_streams=num_residual_streams, **block_kwargs)
                  for _ in range(depth)]
        self.blocks = ModuleList(blocks)
        self.noise_embedder = PatchEmbed(32, 2, 4, 1152, bias=True)
        self.pose_embedder = PatchEmbed(32, 2, 4, 1152, bias=True)
        self.cloth_embedder = PatchEmbed(32, 2, 4, 1152, bias=True)
        norms = [RMSNorm(dim) for dim in dim_modalities]
        self.norms = ModuleList(norms)
        num_patches = self.noise_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, 1152), requires_grad=False)
        self.final_layer = FinalLayer(1152, 2, 4)
        self.t_embedder = TimestepEmbedder(1152)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.noise_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.noise_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.noise_embedder.proj.bias, 0)

        w = self.pose_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.pose_embedder.proj.bias, 0)

        w = self.cloth_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.cloth_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)


        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = 4
        p = self.noise_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(
            self,
            *,
            modality_tokens: list[Tensor, ...],
            modality_masks: tuple[Tensor | None, ...] | None = None,
            time_cond=None
    ):

        modality_tokens[0] = self.noise_embedder(modality_tokens[0]) + self.pos_embed
        modality_tokens[1] = self.pose_embedder(modality_tokens[1]) + self.pos_embed
        modality_tokens[2] = self.cloth_embedder(modality_tokens[2]) + self.pos_embed

        time_cond = self.t_embedder(time_cond)

        modality_tokens = [self.expand_streams(modality) for modality in modality_tokens]

        for block in self.blocks:
            modality_tokens = block(
                time_cond=time_cond,
                modality_tokens=modality_tokens,
                modality_masks=modality_masks
            )

        modality_tokens = [self.reduce_streams(modality) for modality in modality_tokens]
        modality_tokens = [norm(tokens) for tokens, norm in zip(modality_tokens, self.norms)]
        image_tokens = self.final_layer(modality_tokens[0], time_cond)  # (N, T, patch_size ** 2 * out_channels)
        image_tokens = self.unpatchify(image_tokens)
        return image_tokens