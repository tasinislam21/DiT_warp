from __future__ import annotations
from models import get_2d_sincos_pos_embed
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from timm.models.vision_transformer import PatchEmbed
from models import TimestepEmbedder, FinalLayer
from einops import  repeat, pack, unpack
from einops.layers.torch import Rearrange

from x_transformers.attend import Attend
from x_transformers import (
    RMSNorm,
    FeedForward
)

from hyper_connections import (
    HyperConnections,
    Residual
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def softclamp(t, value):
    return (t / value).tanh() * value

# rmsnorm

class MultiHeadRMSNorm(Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

# attention

class JointAttention(Module):
    def __init__(
        self,
        *,
        dim_inputs: tuple[int, ...],
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        flash = False,
        softclamp = False,
        softclamp_value = 50.,
        attend_kwargs: dict = dict()
    ):
        super().__init__()
        """
        ein notation

        b - batch
        h - heads
        n - sequence
        d - feature dimension
        """

        dim_inner = dim_head * heads

        num_inputs = len(dim_inputs)
        self.num_inputs = num_inputs

        self.to_qkv = ModuleList([nn.Linear(dim_input, dim_inner * 3, bias = False) for dim_input in dim_inputs])

        self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)

        self.attend = Attend(
            flash = flash,
            softclamp_logits = softclamp,
            logit_softclamp_value = softclamp_value,
            **attend_kwargs
        )

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = ModuleList([nn.Linear(dim_inner, dim_input, bias = False) for dim_input in dim_inputs])

        self.qk_rmsnorm = qk_rmsnorm
        self.q_rmsnorms = (None,) * num_inputs
        self.k_rmsnorms = (None,) * num_inputs

        if qk_rmsnorm:
            self.q_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])
            self.k_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])

        self.register_buffer('dummy', torch.tensor(0), persistent = False)

    def forward(
        self,
        inputs: tuple[Tensor],
        masks: tuple[Tensor | None] | None = None
    ):

        device = self.dummy.device

        assert len(inputs) == self.num_inputs

        masks = default(masks, (None,) * self.num_inputs)

        # project each modality separately for qkv
        # also handle masks, assume None means attend to all tokens

        all_qkvs = []
        all_masks = []

        for x, mask, to_qkv, q_rmsnorm, k_rmsnorm in zip(inputs, masks, self.to_qkv, self.q_rmsnorms, self.k_rmsnorms):

            qkv = to_qkv(x)
            qkv = self.split_heads(qkv)

            # optional qk rmsnorm per modality

            if self.qk_rmsnorm:
                q, k, v = qkv
                q = q_rmsnorm(q)
                k = k_rmsnorm(k)
                qkv = torch.stack((q, k, v))

            all_qkvs.append(qkv)

            # handle mask per modality

            if not exists(mask):
                mask = torch.ones(x.shape[:2], device = device, dtype = torch.bool)

            all_masks.append(mask)

        # combine all qkv and masks

        all_qkvs, packed_shape = pack(all_qkvs, 'qkv b h * d')
        all_masks, _ = pack(all_masks, 'b *')

        # attention

        q, k, v = all_qkvs

        outs, *_ = self.attend(q, k, v, mask = all_masks)

        # merge heads and then separate by modality for combine heads projection

        outs = self.merge_heads(outs)
        outs = unpack(outs, packed_shape, 'b * d')

        # separate combination of heads for each modality

        all_outs = []

        for out, to_out in zip(outs, self.to_out):
            out = to_out(out)
            all_outs.append(out)

        return tuple(all_outs)

# class

class MMDiTBlock(Module):
    def __init__(
        self,
        *,
        dim_text,
        dim_image,
        dim_cond = None,
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        flash_attn = False,
        num_residual_streams = 1,
        ff_kwargs: dict = dict()
    ):
        super().__init__()

        # residual functions / maybe hyper connections

        residual_klass = Residual if num_residual_streams == 1 else HyperConnections

        self.text_attn_residual_fn = residual_klass(num_residual_streams, dim = dim_text)
        self.text_ff_residual_fn = residual_klass(num_residual_streams, dim = dim_text)

        self.image_attn_residual_fn = residual_klass(num_residual_streams, dim = dim_image)
        self.image_ff_residual_fn = residual_klass(num_residual_streams, dim = dim_image)

        # handle optional time conditioning

        has_cond = exists(dim_cond)
        self.has_cond = has_cond

        if has_cond:
            dim_gammas = (
                *((dim_text,) * 4),
                *((dim_image,) * 4)
            )

            dim_betas = (
                *((dim_text,) * 2),
                *((dim_image,) * 2),
            )

            self.cond_dims = (*dim_gammas, *dim_betas)

            to_cond_linear = nn.Linear(dim_cond, sum(self.cond_dims))

            self.to_cond = nn.Sequential(
                Rearrange('b d -> b 1 d'),
                nn.SiLU(),
                to_cond_linear
            )

            nn.init.zeros_(to_cond_linear.weight)
            nn.init.zeros_(to_cond_linear.bias)
            nn.init.constant_(to_cond_linear.bias[:sum(dim_gammas)], 1.)

        # handle adaptive norms

        self.text_attn_layernorm = nn.LayerNorm(dim_text, elementwise_affine = not has_cond)
        self.image_attn_layernorm = nn.LayerNorm(dim_image, elementwise_affine = not has_cond)

        self.text_ff_layernorm = nn.LayerNorm(dim_text, elementwise_affine = not has_cond)
        self.image_ff_layernorm = nn.LayerNorm(dim_image, elementwise_affine = not has_cond)

        # attention and feedforward

        self.joint_attn = JointAttention(
            dim_inputs = (dim_text, dim_image),
            dim_head = dim_head,
            heads = heads,
            flash = flash_attn
        )

        self.text_ff = FeedForward(dim_text, **ff_kwargs)
        self.image_ff = FeedForward(dim_image, **ff_kwargs)

    def forward(
        self,
        *,
        text_tokens,
        image_tokens,
        text_mask = None,
        time_cond = None,
        skip_feedforward_text_tokens = True
    ):
        assert not (exists(time_cond) ^ self.has_cond), 'time condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

        if self.has_cond:
            (
                text_pre_attn_gamma,
                text_post_attn_gamma,
                text_pre_ff_gamma,
                text_post_ff_gamma,
                image_pre_attn_gamma,
                image_post_attn_gamma,
                image_pre_ff_gamma,
                image_post_ff_gamma,
                text_pre_attn_beta,
                text_pre_ff_beta,
                image_pre_attn_beta,
                image_pre_ff_beta,
            ) = self.to_cond(time_cond).split(self.cond_dims, dim = -1)

        # handle attn adaptive layernorm

        text_tokens, add_text_residual = self.text_attn_residual_fn(text_tokens)
        image_tokens, add_image_residual = self.image_attn_residual_fn(image_tokens)

        text_tokens = self.text_attn_layernorm(text_tokens)
        image_tokens = self.image_attn_layernorm(image_tokens)

        if self.has_cond:
            text_tokens = text_tokens * text_pre_attn_gamma + text_pre_attn_beta
            image_tokens = image_tokens * image_pre_attn_gamma + image_pre_attn_beta

        # attention

        text_tokens, image_tokens = self.joint_attn(
            inputs = (text_tokens, image_tokens),
            masks = (text_mask, None)
        )

        # condition attention output

        if self.has_cond:
            text_tokens = text_tokens * text_post_attn_gamma
            image_tokens = image_tokens * image_post_attn_gamma

        # add attention residual

        text_tokens = add_text_residual(text_tokens)
        image_tokens = add_image_residual(image_tokens)

        # handle feedforward adaptive layernorm

        if not skip_feedforward_text_tokens:
            text_tokens, add_text_residual = self.text_ff_residual_fn(text_tokens)
            text_tokens = self.text_ff_layernorm(text_tokens)

            if self.has_cond:
                text_tokens = text_tokens * text_pre_ff_gamma + text_pre_ff_beta

        image_tokens, add_image_residual = self.image_ff_residual_fn(image_tokens)
        image_tokens = self.image_ff_layernorm(image_tokens)

        if self.has_cond:
            image_tokens = image_tokens * image_pre_ff_gamma + image_pre_ff_beta

        # images feedforward

        image_tokens = self.image_ff(image_tokens)

        # images condition feedforward output

        if self.has_cond:
            image_tokens = image_tokens * image_post_ff_gamma

        # images feedforward residual

        image_tokens = add_image_residual(image_tokens)

        # early return, for last block in mmdit

        if skip_feedforward_text_tokens:
            return text_tokens, image_tokens

        # text feedforward

        text_tokens = self.text_ff(text_tokens)

        # text condition feedforward output

        if self.has_cond:
            text_tokens = text_tokens * text_post_ff_gamma

        # text feedforward residual

        text_tokens = add_text_residual(text_tokens)

        # return

        return text_tokens, image_tokens

# mm dit transformer - simply many blocks

class MMDiT(Module):
    def __init__(
        self,
        *,
        depth,
        dim_image = 1152,
        num_register_tokens = 0,
        final_norm = True,
        num_residual_streams = 4,
        **block_kwargs
    ):
        super().__init__()

        self.expand_streams, self.reduce_streams = HyperConnections.get_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        self.has_register_tokens = num_register_tokens > 0
        self.register_tokens = nn.Parameter(torch.zeros(num_register_tokens, dim_image))
        nn.init.normal_(self.register_tokens, std = 0.02)

        self.blocks = ModuleList([])
        self.noise_embedder = PatchEmbed(32, 2, 4, 1152, bias=True)
        self.pose_embedder = PatchEmbed(32, 2, 4, 1152, bias=True)
        self.cloth_embedder = PatchEmbed(32, 2, 4, 1152, bias=True)

        num_patches = self.noise_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, 1152), requires_grad=False)
        self.t_embedder = TimestepEmbedder(1152)

        for _ in range(depth):
            block = MMDiTBlock(
                dim_image = dim_image,
                num_residual_streams = num_residual_streams,
                **block_kwargs
            )

            self.blocks.append(block)

        self.norm = RMSNorm(dim_image) if final_norm else nn.Identity()
        self.final_layer = FinalLayer(1152, 2, 4)
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
        pose_tokens,
        cloth_tokens,
        image_tokens,
        text_mask = None,
        time_cond = None,
        should_skip_last_feedforward = True
    ):

        if self.has_register_tokens:
            register_tokens = repeat(self.register_tokens, 'n d -> b n d', b = image_tokens.shape[0])
            image_tokens, packed_shape = pack([register_tokens, image_tokens], 'b * d')

        image_tokens = self.noise_embedder(image_tokens) + self.pos_embed
        pose_tokens = self.pose_embedder(pose_tokens) + self.pos_embed
        cloth_tokens = self.cloth_embedder(cloth_tokens) + self.pos_embed

        time_cond = self.t_embedder(time_cond)

        pose_tokens = self.expand_streams(pose_tokens)
        cloth_tokens = self.expand_streams(cloth_tokens)
        image_tokens = self.expand_streams(image_tokens)

        for ind, block in enumerate(self.blocks):
            is_last = ind == (len(self.blocks) - 1)

            text_tokens, image_tokens = block(
                time_cond = time_cond,
                text_tokens = text_tokens,
                image_tokens = image_tokens,
                text_mask = text_mask,
                skip_feedforward_text_tokens = is_last and should_skip_last_feedforward
            )

        if self.has_register_tokens:
            _, image_tokens = unpack(image_tokens, packed_shape, 'b * d')

        image_tokens = self.reduce_streams(image_tokens)
        image_tokens = self.norm(image_tokens)
        image_tokens = self.final_layer(image_tokens, time_cond)                # (N, T, patch_size ** 2 * out_channels)
        image_tokens = self.unpatchify(image_tokens)                   # (N, out_channels, H, W)
        return image_tokens