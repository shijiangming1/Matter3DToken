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

from .budgeted_adaptive_router import BudgetedAdaptiveRouter
from .pointlayers import EmbeddingLayer, MergeHead
from .z_aware_prototype_tokenizer import ZAwarePrototypeTokenizer
from .transformer import Transformer


class Matter3DToken(nn.Module):
    def __init__(
        self,
        emb_cin,
        emb_chidden,
        emb_num_layers,
        vit_dim,
        vit_depth,
        vit_num_heads,
        vit_expansion,
        merge_chidden,
        merge_cout,
        nb_class,
        drop_path=0.0,
        density_encoding=False,
        z_aware_prototypes=None,
        budgeted_adaptive_router=None,
        sparse_batch_reindex=False,
    ):
        super().__init__()

        # Pre-normalization
        self.norm = nn.BatchNorm1d(emb_cin)

        # Embedding layer
        self.embed = self.get_embed(
            emb_num_layers,
            emb_cin,
            emb_chidden,
            vit_dim,
        )

        self.z_aware_prototype_config = z_aware_prototypes or {}
        self.use_z_aware_prototypes = self.z_aware_prototype_config.get("enabled", False)
        self.budgeted_router_config = budgeted_adaptive_router or {}
        self.use_budgeted_adaptive_router = self.budgeted_router_config.get("enabled", False)
        self.z_aware_residual_blend = self.use_z_aware_prototypes and self.z_aware_prototype_config.get(
            "residual_blend", False
        )
        self.router_residual_blend = self.use_budgeted_adaptive_router and self.budgeted_router_config.get(
            "residual_blend", False
        )
        self.use_sparse_batch_reindex = sparse_batch_reindex
        self.joint_blend = nn.Parameter(torch.tensor(0.0))

        # Transformer in BEV
        self.vit = Transformer(
            vit_dim,
            vit_depth,
            vit_num_heads,
            vit_expansion,
            drop_path,
            coord_dim=3 if self.use_z_aware_prototypes else 2,
            z_phase_init=(
                self.z_aware_prototype_config.get("z_phase_init", 0.0)
                if self.use_z_aware_prototypes
                else 0.0
            ),
        )

        if self.use_z_aware_prototypes:
            self.z_aware_tokenizer = ZAwarePrototypeTokenizer(
                vit_dim,
                extra_ratio=self.z_aware_prototype_config.get("extra_ratio", 0.15),
                entropy_bins=self.z_aware_prototype_config.get("entropy_bins", 8),
                min_entropy=self.z_aware_prototype_config.get("min_entropy", 0.15),
                gumbel_tau=self.z_aware_prototype_config.get("gumbel_tau", 1.0),
                preserve_anchor=self.z_aware_prototype_config.get("preserve_anchor", False),
                deterministic_assignment=self.z_aware_prototype_config.get(
                    "deterministic_assignment", False
                ),
                extra_gate_init=self.z_aware_prototype_config.get("extra_gate_init", 1.0),
            )
            if self.z_aware_residual_blend:
                self.z_aware_residual_alpha = nn.Parameter(
                    torch.tensor(
                        float(self.z_aware_prototype_config.get("residual_alpha_init", 0.0))
                    )
                )
        if self.use_budgeted_adaptive_router:
            self.budgeted_adaptive_router = BudgetedAdaptiveRouter(
                vit_dim,
                fine_size=self.budgeted_router_config.get("fine_size", 0.5),
                coarse_factor=self.budgeted_router_config.get("coarse_factor", 2),
                token_ratio=self.budgeted_router_config.get("token_ratio", 0.75),
                sparse_weight=self.budgeted_router_config.get("sparse_weight", 1.0),
                detail_weight=self.budgeted_router_config.get("detail_weight", 0.0),
                semantic_weight=self.budgeted_router_config.get("semantic_weight", 0.0),
                score_noise=self.budgeted_router_config.get("score_noise", 0.1),
                detail_relay=self.budgeted_router_config.get("detail_relay", False),
                detail_relay_all=self.budgeted_router_config.get(
                    "detail_relay_all", False
                ),
            )
            if self.router_residual_blend:
                self.router_residual_alpha = nn.Parameter(
                    torch.tensor(
                        float(self.budgeted_router_config.get("residual_alpha_init", 0.0))
                    )
                )
        self.z_aware_stats = None
        self.budgeted_router_stats = None
        self.sparse_batch_reindex_stats = None

        # Preserve pillar occupancy, which is otherwise discarded by max pooling.
        self.density_encoding = density_encoding
        if density_encoding:
            self.density_embedding = nn.Parameter(torch.zeros(vit_dim))

        # Padding token
        self.pad_token = nn.Parameter(torch.zeros((vit_dim)), requires_grad=True)

        # Point head to merge BEV and point features
        self.merge = MergeHead(vit_dim, merge_chidden, merge_cout)

        # Classif.
        self.classif = nn.Linear(merge_cout, nb_class, bias=True)

        #
        self.init_weights()

    def get_embed(self, emb_num_layers, emb_cin, emb_chidden, vit_dim):
        embed = nn.ModuleList()

        if emb_num_layers > 1:
            channels = [(emb_cin, emb_chidden, emb_chidden)]
            for i in range(1, emb_num_layers - 1):
                channels += [(emb_chidden, emb_chidden, emb_chidden)]
            channels += [(emb_chidden, emb_chidden, vit_dim)]
        else:
            channels = [(emb_cin, emb_chidden, vit_dim)]

        for i, (cin, chidden, cout) in enumerate(channels):
            embed.append(EmbeddingLayer(cin, chidden, cout))
        return embed

    def init_weights(self):
        for m in [self.norm, self.classif]:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _sparse_bev_lift(self, point_emb, xy_bev, idx_to_bev, bev_splits):
        """Run the unmodified sparse Matter3DToken path and return per-point context."""
        channels = point_emb.shape[1]
        total_tokens = int(bev_splits.sum())
        bev_feat = torch.zeros(
            total_tokens,
            channels,
            dtype=point_emb.dtype,
            device=point_emb.device,
        )
        bev_feat.scatter_reduce_(
            0,
            idx_to_bev[:, None].expand(-1, channels),
            point_emb,
            reduce="amax",
            include_self=False,
        )
        offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=bev_splits.device),
                bev_splits.long().cumsum(0),
            )
        )
        coords = xy_bev.to(point_emb.dtype)
        if self.use_z_aware_prototypes:
            coords = torch.cat((coords, torch.zeros_like(coords[:, :1])), dim=1)
        bev_feat = self.vit(bev_feat, coords, offsets)
        return bev_feat[idx_to_bev]

    def forward(
        self,
        feats,
        xyz,
        neighbors,
        xy_bev,
        idx_to_bev,
        idx_occ,
        idx_pad,
        bev_splits,
    ):

        # Pre-normalization
        point_emb = self.norm(feats)

        # Point embedding
        for embed in self.embed:
            point_emb = embed(point_emb, neighbors)

        # Dimension: batch_size x channels x sequence length
        B, C = len(bev_splits), point_emb.shape[1]
        pillar_layout = (
            bev_splits
            if self.use_sparse_batch_reindex
            else int(bev_splits[0].item())
        )
        M = pillar_layout

        if self.use_z_aware_prototypes and self.use_budgeted_adaptive_router:
            z_out = self.z_aware_tokenizer(
                point_emb, xyz, idx_to_bev, xy_bev, idx_occ, M, B,
                self.pad_token, pack_tokens=True,
            )
            a_out = self.budgeted_adaptive_router(
                point_emb, xyz, idx_to_bev, xy_bev, idx_occ, M, B,
                self.pad_token, pack_tokens=True,
            )
            z_tokens, z_coords, z_base, z_extra, z_weights, z_stats, z_offsets = z_out
            a_tokens, a_coords, a_context, a_fine, a_detail, a_stats, a_offsets = a_out
            token_parts = []
            coord_parts = []
            joint_offsets = [0]
            z_base_joint, z_extra_joint, a_context_joint = [], [], []
            a_fine_joint = [] if a_fine is not None else None
            point_batch = torch.bucketize(idx_to_bev, bev_splits.cumsum(0)[1:])
            for batch_idx in range(B):
                z0, z1 = int(z_offsets[batch_idx]), int(z_offsets[batch_idx + 1])
                a0, a1 = int(a_offsets[batch_idx]), int(a_offsets[batch_idx + 1])
                offset = joint_offsets[-1]
                token_parts.extend((z_tokens[z0:z1], a_tokens[a0:a1]))
                coord_parts.extend((z_coords[z0:z1], torch.cat(
                    (a_coords[a0:a1], torch.zeros(a1 - a0, 1,
                    dtype=a_coords.dtype, device=a_coords.device)), dim=1)))
                joint_offsets.append(offset + (z1 - z0) + (a1 - a0))
            joint_tokens = torch.cat(token_parts, dim=0)
            joint_coords = torch.cat(coord_parts, dim=0)
            joint_offsets = torch.tensor(joint_offsets, dtype=torch.long,
                                         device=point_emb.device)
            context_cursor = 0
            for batch_idx in range(B):
                z_shift = int(joint_offsets[batch_idx])
                a_shift = z_shift + int(z_offsets[batch_idx + 1] - z_offsets[batch_idx])
                mask = point_batch == batch_idx
                z_base_joint.append(z_base[mask] - int(z_offsets[batch_idx]) + z_shift)
                z_extra_joint.append(z_extra[mask] - int(z_offsets[batch_idx]) + z_shift)
                occupied_count = int(point_batch[point_batch == batch_idx].numel())
                if a_detail is not None:
                    occupied_count = int(torch.unique(idx_to_bev[mask]).numel())
                a_context_joint.append(a_context[context_cursor:context_cursor + occupied_count] + a_shift)
                context_cursor += occupied_count
                if a_fine_joint is not None:
                    a_fine_joint.append(a_fine[mask])
            z_base_joint = torch.cat(z_base_joint)
            z_extra_joint = torch.cat(z_extra_joint)
            a_context_joint = torch.cat(a_context_joint)
            if a_fine_joint is not None:
                a_fine_joint = torch.cat(a_fine_joint)
            shared = self.vit(joint_tokens, joint_coords, joint_offsets)
            z_lift = shared[z_base_joint] * z_weights[:, :1] + shared[z_extra_joint] * z_weights[:, 1:2]
            if a_detail is None:
                a_lift = shared[a_context_joint]
            else:
                a_lift = (shared[a_context_joint] + a_detail)[a_fine_joint]
            blend = self.joint_blend.sigmoid()
            lifted = blend * z_lift + (1.0 - blend) * a_lift
            self.z_aware_stats = z_stats
            self.budgeted_router_stats = a_stats
            self.sparse_batch_reindex_stats = {
                "tokens": int(joint_tokens.shape[0]),
                "z_tokens": z_stats["tokens"],
                "routed_tokens": a_stats["tokens"],
                "padded_tokens": 0,
                "padding_removed": 0,
            }
            point_emb = self.merge(point_emb, lifted)
            return self.classif(point_emb)

        if self.use_z_aware_prototypes:
            baseline_lifted = (
                self._sparse_bev_lift(point_emb, xy_bev, idx_to_bev, bev_splits)
                if self.z_aware_residual_blend
                else None
            )
            (
                bev_feat,
                rope_coords,
                base_route,
                extra_route,
                route_weights,
                self.z_aware_stats,
                token_offsets,
            ) = self.z_aware_tokenizer(
                point_emb,
                xyz,
                idx_to_bev,
                xy_bev,
                idx_occ,
                M,
                B,
                self.pad_token,
                pack_tokens=self.use_sparse_batch_reindex,
            )
            bev_feat = self.vit(
                bev_feat,
                rope_coords,
                token_offsets if self.use_sparse_batch_reindex else None,
            )
            flat_bev = bev_feat if self.use_sparse_batch_reindex else bev_feat.reshape(-1, C)
            if self.use_sparse_batch_reindex:
                padded_tokens = len(bev_splits) * int(
                    (token_offsets[1:] - token_offsets[:-1]).max()
                )
                self.sparse_batch_reindex_stats = {
                    "tokens": self.z_aware_stats["tokens"],
                    "padded_tokens": padded_tokens,
                    "padding_removed": padded_tokens - self.z_aware_stats["tokens"],
                }
            lifted = (
                flat_bev[base_route] * route_weights[:, 0:1]
                + flat_bev[extra_route] * route_weights[:, 1:2]
            )
            if baseline_lifted is not None:
                lifted = baseline_lifted + self.z_aware_residual_alpha.to(
                    lifted.dtype
                ) * (lifted - baseline_lifted)
            point_emb = self.merge(point_emb, lifted)
            return self.classif(point_emb)

        if self.use_budgeted_adaptive_router:
            baseline_lifted = (
                self._sparse_bev_lift(point_emb, xy_bev, idx_to_bev, bev_splits)
                if self.router_residual_blend
                else None
            )
            (
                bev_feat,
                rope_coords,
                context_route,
                point_fine_route,
                fine_detail,
                self.budgeted_router_stats,
                token_offsets,
            ) = (
                self.budgeted_adaptive_router(
                    point_emb,
                    xyz,
                    idx_to_bev,
                    xy_bev,
                    idx_occ,
                    pillar_layout,
                    B,
                    self.pad_token,
                    pack_tokens=self.use_sparse_batch_reindex,
                )
            )
            bev_feat = self.vit(
                bev_feat,
                rope_coords,
                token_offsets if self.use_sparse_batch_reindex else None,
            )
            flat_bev = bev_feat if self.use_sparse_batch_reindex else bev_feat.reshape(-1, C)
            if self.use_sparse_batch_reindex:
                self.sparse_batch_reindex_stats = {
                    "tokens": self.budgeted_router_stats["tokens"],
                    "padded_tokens": self.budgeted_router_stats["padded_tokens"],
                    "padding_removed": self.budgeted_router_stats["padding_removed"],
                }
            if fine_detail is None:
                lifted = flat_bev[context_route]
            else:
                fine_context = flat_bev[context_route] + fine_detail
                lifted = fine_context[point_fine_route]
            if baseline_lifted is not None:
                lifted = baseline_lifted + self.router_residual_alpha.to(
                    lifted.dtype
                ) * (lifted - baseline_lifted)
            point_emb = self.merge(point_emb, lifted)
            return self.classif(point_emb)

        if self.use_sparse_batch_reindex:
            total_tokens = int(bev_splits.sum())
            bev_feat = torch.zeros(
                total_tokens,
                C,
                dtype=point_emb.dtype,
                device=point_emb.device,
            )
            bev_feat.scatter_reduce_(
                0,
                idx_to_bev[:, None].expand(-1, C),
                point_emb,
                reduce="amax",
                include_self=False,
            )
            offsets = torch.cat(
                (
                    torch.zeros(1, dtype=torch.long, device=bev_splits.device),
                    bev_splits.long().cumsum(0),
                )
            )
            bev_feat = self.vit(
                bev_feat,
                xy_bev.to(point_emb.dtype),
                offsets,
            )
            padded_tokens = len(bev_splits) * int(bev_splits.max())
            self.sparse_batch_reindex_stats = {
                "tokens": total_tokens,
                "padded_tokens": padded_tokens,
                "padding_removed": padded_tokens - total_tokens,
            }
            point_emb = self.merge(point_emb, bev_feat[idx_to_bev])
            return self.classif(point_emb)

        # Init BEV map with zeros
        bev_feat = torch.zeros(
            (B * M, C), dtype=point_emb.dtype, device=point_emb.device
        )

        # Actual projection
        idx_bev_expand = idx_to_bev.unsqueeze(1).expand(-1, C)
        bev_feat.scatter_reduce_(
            0, idx_bev_expand, point_emb, reduce="amax", include_self=False
        )

        if self.density_encoding:
            density = torch.zeros(
                (B * M, 1), dtype=point_emb.dtype, device=point_emb.device
            )
            density.scatter_add_(
                0,
                idx_to_bev.unsqueeze(1),
                torch.ones(
                    (len(idx_to_bev), 1),
                    dtype=point_emb.dtype,
                    device=point_emb.device,
                ),
            )
            density = torch.log1p(density).reshape(B, M, 1)
            density = density / density.amax(dim=1, keepdim=True).clamp_min(1e-6)
            density = density.reshape(B * M, 1)
            density_embedding = self.density_embedding.to(point_emb.dtype)
            density_residual = (density * density_embedding).to(bev_feat.dtype)
            bev_feat = bev_feat + density_residual

        # Padding token
        if self.training:
            bev_feat = bev_feat.clone()
        if idx_pad is not None:
            pad_token = (
                self.pad_token.to(point_emb.dtype).unsqueeze(0).expand(len(idx_pad), -1)
            )
            bev_feat.index_add_(0, idx_pad, pad_token)

        # Positional encoding
        rope_coords = torch.zeros(
            (B * M, 2), dtype=point_emb.dtype, device=point_emb.device
        )
        rope_coords[idx_occ] = xy_bev.to(point_emb.dtype)
        rope_coords = rope_coords.reshape(B, M, 2)

        # Transformer in BEV
        bev_feat = bev_feat.reshape(B, M, -1)
        bev_feat = self.vit(bev_feat, rope_coords)

        # Lift bev feat and merge with point feat
        point_emb = self.merge(point_emb, bev_feat.reshape(B * M, C)[idx_to_bev])

        # Classif
        return self.classif(point_emb)
