import math

import torch
import torch.nn as nn


class BudgetedAdaptiveRouter(nn.Module):
    """Build one budgeted token partition from fine BEV pillar candidates."""

    def __init__(
        self,
        dim,
        fine_size=0.5,
        coarse_factor=2,
        token_ratio=0.75,
        sparse_weight=1.0,
        detail_weight=0.0,
        semantic_weight=0.0,
        score_noise=0.1,
        detail_relay=False,
        detail_relay_all=False,
    ):
        super().__init__()
        self.fine_size = fine_size
        self.coarse_size = fine_size * coarse_factor
        self.token_ratio = token_ratio
        self.sparse_weight = sparse_weight
        self.detail_weight = detail_weight
        self.semantic_weight = semantic_weight
        self.score_noise = score_noise
        self.detail_relay = detail_relay
        self.detail_relay_all = detail_relay_all

        hidden = max(dim // 4, 32)
        self.semantic_score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.semantic_score[-1].weight)
        nn.init.zeros_(self.semantic_score[-1].bias)

        if detail_relay:
            self.detail_projection = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim, bias=False),
            )
            self.offset_projection = nn.Linear(2, dim, bias=False)
            nn.init.zeros_(self.detail_projection[-1].weight)
            nn.init.zeros_(self.offset_projection.weight)

    @staticmethod
    def _pool(values, index, size, reduce="amax"):
        shape = (size,) + values.shape[1:]
        if reduce == "amax":
            out = torch.zeros(shape, dtype=values.dtype, device=values.device)
            expanded = index.reshape((-1,) + (1,) * (values.ndim - 1)).expand_as(values)
            out.scatter_reduce_(0, expanded, values, reduce="amax", include_self=False)
            return out
        out = torch.zeros(shape, dtype=values.dtype, device=values.device)
        expanded = index.reshape((-1,) + (1,) * (values.ndim - 1)).expand_as(values)
        out.scatter_add_(0, expanded, values)
        return out

    def _select_groups(self, score, child_count, target_tokens):
        base_tokens = len(child_count)
        remaining = max(target_tokens - base_tokens, 0)
        selected = torch.zeros(len(child_count), dtype=torch.bool, device=score.device)
        if remaining == 0:
            return selected
        order = score.argsort(descending=True)
        ordered_cost = (child_count[order] - 1).clamp_min(0)
        keep = ordered_cost.cumsum(0) <= remaining
        selected[order[keep & ordered_cost.gt(0)]] = True
        return selected

    @staticmethod
    def _standardize(values):
        """Make selector terms comparable within one scan."""
        values = values.float()
        centered = values - values.mean()
        return centered / centered.square().mean().sqrt().clamp_min(1e-3)

    def forward(
        self,
        point_features,
        point_xyz,
        point_to_pillar,
        pillar_xy,
        occupied_pillars,
        pillars_per_sample,
        batch_size,
        pad_token,
        pack_tokens=False,
    ):
        channels = point_features.shape[1]
        if torch.is_tensor(pillars_per_sample):
            pillar_lengths = pillars_per_sample.long()
        else:
            pillar_lengths = torch.full(
                (batch_size,),
                int(pillars_per_sample),
                dtype=torch.long,
                device=point_features.device,
            )
        pillar_offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=point_features.device),
                pillar_lengths.cumsum(0),
            )
        )
        num_pillars = int(pillar_offsets[-1])
        expanded_idx = point_to_pillar[:, None].expand(-1, channels)

        fine_features = torch.zeros(
            num_pillars,
            channels,
            dtype=point_features.dtype,
            device=point_features.device,
        )
        fine_features.scatter_reduce_(
            0, expanded_idx, point_features, reduce="amax", include_self=False
        )

        counts = torch.zeros(num_pillars, dtype=torch.float32, device=point_xyz.device)
        counts.scatter_add_(0, point_to_pillar, torch.ones_like(point_xyz[:, 2]).float())
        z_sum = torch.zeros_like(counts)
        z_sq_sum = torch.zeros_like(counts)
        z = point_xyz[:, 2].float()
        z_sum.scatter_add_(0, point_to_pillar, z)
        z_sq_sum.scatter_add_(0, point_to_pillar, z.square())

        global_xy = torch.zeros(
            num_pillars, 2, dtype=point_features.dtype, device=point_features.device
        )
        global_xy[occupied_pillars.long()] = pillar_xy.to(point_features.dtype)

        sample_tokens = []
        sample_coords = []
        route = torch.empty(num_pillars, dtype=torch.long, device=point_features.device)
        fine_total = 0
        coarse_total = 0
        split_total = 0
        geometry_values = []
        heterogeneity_values = []
        selected_density = []
        merged_density = []
        detail_features = []
        occupied_counts = []
        packed_fine_route = None
        if self.detail_relay:
            packed_fine_route = torch.empty(
                num_pillars, dtype=torch.long, device=point_features.device
            )
        packed_fine_offset = 0

        for batch_idx in range(batch_size):
            start = int(pillar_offsets[batch_idx])
            end = int(pillar_offsets[batch_idx + 1])
            occupied = int(counts[start:end].gt(0).sum())
            occupied_counts.append(occupied)
            fine_total += occupied
            fine_global = torch.arange(start, start + occupied, device=point_features.device)
            fine_xy = global_xy[fine_global]
            parent_coords = torch.floor(fine_xy.float() / self.coarse_size).long()
            unique_parent, fine_to_parent = torch.unique(
                parent_coords, dim=0, sorted=True, return_inverse=True
            )
            num_parent = len(unique_parent)
            coarse_total += num_parent

            fine_feat = fine_features[fine_global]
            parent_feat = self._pool(fine_feat, fine_to_parent, num_parent)
            child_count = torch.bincount(fine_to_parent, minlength=num_parent)
            parent_count = self._pool(
                counts[fine_global, None], fine_to_parent, num_parent, reduce="sum"
            )[:, 0]
            parent_z_sum = self._pool(
                z_sum[fine_global, None], fine_to_parent, num_parent, reduce="sum"
            )[:, 0]
            parent_z_sq_sum = self._pool(
                z_sq_sum[fine_global, None], fine_to_parent, num_parent, reduce="sum"
            )[:, 0]
            parent_z_mean = parent_z_sum / parent_count.clamp_min(1)
            parent_z_std = (
                parent_z_sq_sum / parent_count.clamp_min(1) - parent_z_mean.square()
            ).clamp_min(0).sqrt()

            geometry = self._standardize(parent_z_std)
            inverse_fine_density = counts[fine_global].clamp_min(1).rsqrt()
            sparse_detail = self._pool(
                inverse_fine_density[:, None],
                fine_to_parent,
                num_parent,
            )[:, 0]
            sparse_detail = self._standardize(sparse_detail)
            if self.detail_weight != 0 or self.detail_relay:
                parent_mean = self._pool(
                    fine_feat, fine_to_parent, num_parent, reduce="sum"
                ) / child_count[:, None].clamp_min(1)
                child_deviation = (
                    fine_feat - parent_mean[fine_to_parent]
                ).square().mean(dim=1).clamp_min(1e-8).sqrt()
                heterogeneity = self._pool(
                    child_deviation[:, None], fine_to_parent, num_parent
                )[:, 0].float()
                heterogeneity = self._standardize(heterogeneity)
            else:
                heterogeneity = torch.zeros_like(geometry)
            semantic = self.semantic_score(parent_feat).float()[:, 0]
            score = (
                geometry
                + self.sparse_weight * sparse_detail
                + self.detail_weight * heterogeneity
                + self.semantic_weight * semantic
            )
            if self.training and self.score_noise > 0:
                uniform = torch.rand_like(score).clamp_(1e-6, 1 - 1e-6)
                score = score - torch.log(-torch.log(uniform)) * self.score_noise

            target = int(math.ceil(occupied * self.token_ratio))
            target = min(occupied, max(num_parent, target))
            selected = self._select_groups(score, child_count, target)
            split_total += int(selected.sum())
            selected_fine_mask = selected[fine_to_parent]
            selected_density.append(counts[fine_global][selected_fine_mask])
            merged_density.append(counts[fine_global][~selected_fine_mask])

            unsplit_parent = torch.nonzero(~selected, as_tuple=False).flatten()
            split_fine = torch.nonzero(
                selected[fine_to_parent], as_tuple=False
            ).flatten()
            probability = score[fine_to_parent[split_fine]].sigmoid().to(fine_feat.dtype)
            straight_through = 1 + probability - probability.detach()
            split_features = parent_feat[fine_to_parent[split_fine]] + straight_through[:, None] * (
                fine_feat[split_fine] - parent_feat[fine_to_parent[split_fine]]
            )
            tokens = torch.cat((parent_feat[unsplit_parent], split_features), dim=0)
            parent_xy = (
                unique_parent[unsplit_parent].to(fine_xy.dtype) * self.coarse_size
                + self.coarse_size / 2
            )
            coords = torch.cat((parent_xy, fine_xy[split_fine]), dim=0)

            parent_route = torch.full(
                (num_parent,), -1, dtype=torch.long, device=point_features.device
            )
            parent_route[unsplit_parent] = torch.arange(
                len(unsplit_parent), device=point_features.device
            )
            local_route = parent_route[fine_to_parent]
            local_route[split_fine] = torch.arange(
                len(unsplit_parent), len(tokens), device=point_features.device
            )

            if self.detail_relay:
                parent_center = (
                    unique_parent[fine_to_parent].to(fine_xy.dtype)
                    * self.coarse_size
                    + self.coarse_size / 2
                )
                relative_xy = (fine_xy - parent_center) / self.coarse_size
                # Use the child-vs-parent-mean residual rather than the
                # max-pooled residual.  This preserves signed boundary
                # information after several fine pillars share one context
                # token, while the zero initialization keeps the module a
                # safe residual at the beginning of training.
                relay = self.detail_projection(
                    fine_feat - parent_mean[fine_to_parent]
                ) + self.offset_projection(relative_xy.to(fine_feat.dtype))
                if not self.detail_relay_all:
                    relay = relay * (~selected_fine_mask).to(relay.dtype)[:, None]

            sample_tokens.append(tokens)
            sample_coords.append(coords)
            route[fine_global] = local_route
            geometry_values.append(parent_z_std.mean())
            heterogeneity_values.append(heterogeneity.mean())
            if self.detail_relay:
                detail_features.append(relay)
                packed_fine_route[fine_global] = torch.arange(
                    packed_fine_offset,
                    packed_fine_offset + occupied,
                    device=point_features.device,
                )
                packed_fine_offset += occupied

        token_lengths = [len(tokens) for tokens in sample_tokens]
        token_offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=point_features.device),
                torch.tensor(
                    token_lengths, dtype=torch.long, device=point_features.device
                ).cumsum(0),
            )
        )
        max_tokens = max(token_lengths)
        if pack_tokens:
            tokens = torch.cat(sample_tokens, dim=0)
            coords = torch.cat(sample_coords, dim=0)
        else:
            tokens = pad_token.to(point_features.dtype).reshape(1, 1, channels).expand(
                batch_size, max_tokens, channels
            ).clone()
            coords = torch.zeros(
                batch_size,
                max_tokens,
                2,
                dtype=point_features.dtype,
                device=point_features.device,
            )
            for batch_idx, (sample_token, sample_coord) in enumerate(
                zip(sample_tokens, sample_coords)
            ):
                length = len(sample_token)
                tokens[batch_idx, :length] = sample_token
                coords[batch_idx, :length] = sample_coord

        if self.detail_relay:
            route_chunks = []
            for batch_idx, token_count in enumerate(occupied_counts):
                pillar_start = int(pillar_offsets[batch_idx])
                local_route = route[
                    pillar_start : pillar_start + token_count
                ]
                route_chunks.append(
                    local_route
                    + (token_offsets[batch_idx] if pack_tokens else batch_idx * max_tokens)
                )
            context_route = torch.cat(route_chunks)
            point_fine_route = packed_fine_route[point_to_pillar]
            detail = torch.cat(detail_features)
        else:
            if pack_tokens:
                point_batch = torch.bucketize(point_to_pillar, pillar_offsets[1:])
                context_route = token_offsets[point_batch] + route[point_to_pillar]
            else:
                point_batch = torch.div(
                    point_to_pillar, pillar_lengths[0], rounding_mode="floor"
                )
                context_route = point_batch * max_tokens + route[point_to_pillar]
            point_fine_route = None
            detail = None
        stats = {
            "tokens": sum(token_lengths),
            "fine_tokens": fine_total,
            "coarse_tokens": coarse_total,
            "split_regions": split_total,
            "mean_vertical_std": torch.stack(geometry_values).mean().detach(),
            "mean_feature_heterogeneity": torch.stack(
                heterogeneity_values
            ).mean().detach(),
            "selected_fine_density": torch.cat(selected_density).mean().detach(),
            "merged_fine_density": torch.cat(merged_density).mean().detach(),
            "padded_tokens": batch_size * max_tokens,
            "padding_removed": batch_size * max_tokens - sum(token_lengths),
        }
        return (
            tokens,
            coords,
            context_route,
            point_fine_route,
            detail,
            stats,
            token_offsets if pack_tokens else None,
        )
