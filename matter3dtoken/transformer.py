# Copyright 2026 - Valeo Comfort and Driving Assistance - valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def apply_nd_rope(q, k, coords, inv_freq, axis_dims, z_phase=None):
    # q, k: batch_size x num_heads x sequence_length x dim_head
    # coords: (x, y) of size batch_size x sequence_length x 2

    q_axes, k_axes = [], []
    start = 0
    for axis, axis_dim in enumerate(axis_dims):
        end = start + axis_dim
        frequency = inv_freq[: axis_dim // 2]
        angle = coords[..., axis : axis + 1].unsqueeze(1) * frequency
        q_axis = rotate(q[..., start:end], angle.cos(), angle.sin())
        k_axis = rotate(k[..., start:end], angle.cos(), angle.sin())
        if z_phase is not None:
            z_angle = coords[..., 2:3].unsqueeze(1) * frequency * z_phase[axis]
            q_axis = rotate(q_axis, z_angle.cos(), z_angle.sin())
            k_axis = rotate(k_axis, z_angle.cos(), z_angle.sin())
        q_axes.append(q_axis)
        k_axes.append(k_axis)
        start = end
    return torch.cat(q_axes, dim=-1), torch.cat(k_axes, dim=-1)


def apply_packed_nd_rope(q, k, coords, inv_freq, axis_dims, z_phase=None):
    # q, k: valid_tokens x num_heads x dim_head
    q_axes, k_axes = [], []
    start = 0
    for axis, axis_dim in enumerate(axis_dims):
        end = start + axis_dim
        frequency = inv_freq[: axis_dim // 2]
        angle = coords[:, axis : axis + 1, None] * frequency
        q_axis = rotate(q[..., start:end], angle.cos(), angle.sin())
        k_axis = rotate(k[..., start:end], angle.cos(), angle.sin())
        if z_phase is not None:
            z_angle = coords[:, 2:3, None] * frequency * z_phase[axis]
            q_axis = rotate(q_axis, z_angle.cos(), z_angle.sin())
            k_axis = rotate(k_axis, z_angle.cos(), z_angle.sin())
        q_axes.append(q_axis)
        k_axes.append(k_axis)
        start = end
    return torch.cat(q_axes, dim=-1), torch.cat(k_axes, dim=-1)


class DropPath(nn.Module):
    def __init__(self, fn, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1.0 - drop_prob
        self.scale = 1.0 / self.keep_prob

        # Function on which to apply droppath
        self.fn = fn

    def extra_repr(self):
        return f"prob={self.drop_prob:.3f}"

    def forward(self, x, coords=None, offsets=None):
        if offsets is not None:
            residual = self.fn(x, coords, offsets).to(x.dtype)
            if not self.training or self.drop_prob == 0.0:
                return x + residual
            batch_size = len(offsets) - 1
            keep_batch = torch.bernoulli(
                torch.full((batch_size,), self.keep_prob, device=x.device)
            ).to(x.dtype)
            sample_id = torch.repeat_interleave(
                torch.arange(batch_size, device=x.device), offsets[1:] - offsets[:-1]
            )
            return x + residual * keep_batch[sample_id, None] * self.scale

        if not self.training or self.drop_prob == 0.0:
            return x + self.fn(x, coords)

        # Mask batch entries
        batch_size = x.shape[0]
        keep_batch = torch.bernoulli(
            torch.full((batch_size,), self.keep_prob, device=x.device)
        ).bool()

        # Drop everything
        if not keep_batch.any():
            return x

        # Efficient droppath
        idx_keep = torch.nonzero(keep_batch).flatten()
        if coords is not None:
            coords = coords[idx_keep]
        residual = self.fn(x[idx_keep], coords).to(x.dtype)

        # Add residual
        out = x.clone()
        out.index_add_(0, idx_keep, residual, alpha=self.scale)

        return out


class ChannelMix(nn.Module):
    def __init__(
        self,
        dim,
        expansion,
        drop_path_prob,
        layerscale_init=1e-5,
    ):
        super().__init__()

        # Pre-norm
        self.norm = nn.LayerNorm(dim)

        # MLP
        hidden_dim = int(dim * expansion)
        self.l1 = nn.Linear(dim, hidden_dim, bias=False)
        self.act = nn.GELU()
        self.l2 = nn.Linear(hidden_dim, dim, bias=False)

        # LayerScale
        self.gamma = nn.Parameter(
            layerscale_init * torch.ones((dim)),
            requires_grad=True,
        )

        # DropPath wrapper
        self.drop_path = DropPath(self._forward_, drop_path_prob)

        # Init.
        self.init_weights()

    def init_weights(self):
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.trunc_normal_(self.l1.weight, std=0.02)
        nn.init.trunc_normal_(self.l2.weight, std=0.02)

    def _forward_(self, x, *args, **kwargs):
        x = self.norm(x)
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        return x * self.gamma

    def forward(self, tokens, offsets=None):
        # Drop path calls _forward_
        return self.drop_path(tokens, offsets=offsets)


class SpatialMix(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        drop_path_prob,
        layerscale_init=1e-5,
        rope_freq=10000,
        coord_dim=2,
        z_phase_init=0.0,
    ):
        super().__init__()

        # Dimensions
        self.num_heads = num_heads
        self.dim_head = dim_head = dim // num_heads
        self.scale = dim_head**-0.5

        # Pre-norm
        self.norm = nn.LayerNorm(dim)

        # Attention
        self.qkv = nn.Linear(dim, 3 * num_heads * dim_head, bias=False)
        self.proj = nn.Linear(num_heads * dim_head, dim, bias=False)

        # RoPE. Keep the pretrained XY channel partition when Z is enabled;
        # height is a zero-initialized residual phase below. Repartitioning
        # 2D RoPE into three axes changes every XY frequency.
        rope_axes = 2
        pairs = dim_head // 2
        pairs_per_axis = [pairs // rope_axes for _ in range(rope_axes)]
        for axis in range(pairs % rope_axes):
            pairs_per_axis[axis] += 1
        self.axis_dims = [2 * value for value in pairs_per_axis]
        max_axis_dim = max(self.axis_dims)
        self.register_buffer(
            "inv_freq",
            1.0
            / (
                rope_freq
                ** (torch.arange(0, max_axis_dim, 2).float() / max_axis_dim)
            ),
        )
        self.z_phase = (
            nn.Parameter(torch.full((rope_axes,), float(z_phase_init)))
            if coord_dim == 3
            else None
        )

        # LayerScale
        self.gamma = nn.Parameter(
            layerscale_init * torch.ones((dim)),
            requires_grad=True,
        )

        # DropPath wrapper
        self.drop_path = DropPath(self._forward_, drop_path_prob)

        # Init.
        self.init_weights()

    def init_weights(self):
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def _forward_(self, x, coords, offsets=None):
        if offsets is not None:
            return self._forward_packed(x, coords, offsets)

        # Shape
        B, N, C = x.shape

        # Norm
        x_norm = self.norm(x)

        # Extract qkv
        qkv = self.qkv(x_norm).view(B, N, 3, self.num_heads, self.dim_head)
        q, k, v = torch.unbind(qkv, dim=2)

        # B x num_heads x sequence_length x dim_head
        q, k, v = [t.transpose(1, 2).contiguous() for t in [q, k, v]]

        # PE
        q, k = apply_nd_rope(
            q, k, coords, self.inv_freq, self.axis_dims, self.z_phase
        )

        # Attention
        x_attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            scale=self.scale,
        )

        # Projection
        x_attn = x_attn.transpose(1, 2)
        x_attn = x_attn.reshape(B, N, self.num_heads * self.dim_head)
        out = self.proj(x_attn)

        # Layerscale
        return out * self.gamma

    def _forward_packed(self, x, coords, offsets):
        num_tokens, channels = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).view(
            num_tokens, 3, self.num_heads, self.dim_head
        )
        q, k, v = torch.unbind(qkv, dim=1)
        q, k = apply_packed_nd_rope(
            q, k, coords, self.inv_freq, self.axis_dims, self.z_phase
        )
        q, k = q.to(v.dtype), k.to(v.dtype)

        q_nested = torch.nested.nested_tensor_from_jagged(
            q.contiguous(), offsets
        ).transpose(1, 2)
        k_nested = torch.nested.nested_tensor_from_jagged(
            k.contiguous(), offsets
        ).transpose(1, 2)
        v_nested = torch.nested.nested_tensor_from_jagged(
            v.contiguous(), offsets
        ).transpose(1, 2)
        x_attn = F.scaled_dot_product_attention(
            q_nested,
            k_nested,
            v_nested,
            dropout_p=0.0,
            scale=self.scale,
        )
        x_attn = x_attn.transpose(1, 2).values().reshape(num_tokens, channels)
        return self.proj(x_attn) * self.gamma

    def forward(self, tokens, coords, offsets=None):
        # Drop path calls compute_attention
        return self.drop_path(tokens, coords, offsets)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        expansion,
        drop_path,
        coord_dim=2,
        z_phase_init=0.0,
    ):
        super().__init__()

        self.channel_mix = nn.ModuleList(
            [ChannelMix(dim, expansion, drop_path) for i in range(depth)]
        )

        self.spatial_mix = nn.ModuleList(
            [
                SpatialMix(
                    dim,
                    num_heads,
                    drop_path,
                    coord_dim=coord_dim,
                    z_phase_init=z_phase_init,
                )
                for i in range(depth)
            ]
        )

    def forward(self, tokens, rope_coords, offsets=None):
        for i, (smix, cmix) in enumerate(zip(self.spatial_mix, self.channel_mix)):
            tokens = smix(tokens, rope_coords, offsets)
            tokens = cmix(tokens, offsets)
        return tokens
