import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZAwarePrototypeTokenizer(nn.Module):
    def __init__(
        self,
        dim,
        extra_ratio=0.15,
        entropy_bins=8,
        min_entropy=0.15,
        gumbel_tau=1.0,
        preserve_anchor=False,
        deterministic_assignment=False,
        extra_gate_init=1.0,
    ):
        super().__init__()
        self.extra_ratio = extra_ratio
        self.entropy_bins = entropy_bins
        self.min_entropy = min_entropy
        self.gumbel_tau = gumbel_tau
        self.preserve_anchor = preserve_anchor
        self.deterministic_assignment = deterministic_assignment
        extra_gate_init = float(extra_gate_init)
        if extra_gate_init < 1.0:
            extra_gate_init = min(max(extra_gate_init, 1e-4), 1.0 - 1e-4)
            self.extra_gate = nn.Parameter(
                torch.tensor(math.log(extra_gate_init / (1.0 - extra_gate_init)))
            )
        else:
            self.register_parameter("extra_gate", None)

        self.assignment = nn.Linear(dim + 1, 2, bias=True)
        nn.init.zeros_(self.assignment.weight)
        nn.init.zeros_(self.assignment.bias)

    def _vertical_statistics(self, z, pillar_idx, num_pillars):
        dtype, device = z.dtype, z.device
        count = torch.zeros(num_pillars, 1, dtype=dtype, device=device)
        count.scatter_add_(0, pillar_idx[:, None], torch.ones_like(z[:, None]))

        z_sum = torch.zeros_like(count)
        z_sum.scatter_add_(0, pillar_idx[:, None], z[:, None])
        mean = z_sum / count.clamp_min(1)

        z_sq_sum = torch.zeros_like(count)
        z_sq_sum.scatter_add_(0, pillar_idx[:, None], z.square()[:, None])
        variance = (z_sq_sum / count.clamp_min(1) - mean.square()).clamp_min(0)
        std = variance.sqrt()

        z_min = torch.full((num_pillars,), torch.inf, dtype=dtype, device=device)
        z_max = torch.full((num_pillars,), -torch.inf, dtype=dtype, device=device)
        z_min.scatter_reduce_(0, pillar_idx, z, reduce="amin", include_self=True)
        z_max.scatter_reduce_(0, pillar_idx, z, reduce="amax", include_self=True)
        z_range = (z_max - z_min).clamp_min(1e-6)

        relative = ((z - z_min[pillar_idx]) / z_range[pillar_idx]).clamp(0, 1)
        bins = (relative * self.entropy_bins).long().clamp_max(self.entropy_bins - 1)
        histogram_idx = pillar_idx * self.entropy_bins + bins
        histogram = torch.zeros(
            num_pillars * self.entropy_bins, dtype=dtype, device=device
        )
        histogram.scatter_add_(0, histogram_idx, torch.ones_like(z))
        histogram = histogram.reshape(num_pillars, self.entropy_bins)
        probability = histogram / count.clamp_min(1)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=1)
        entropy = entropy / math.log(self.entropy_bins)

        normalized_z = (z - mean[pillar_idx, 0]) / std[pillar_idx, 0].clamp_min(1e-3)
        return count[:, 0], mean[:, 0], std[:, 0], entropy, normalized_z

    def _select_split_pillars(self, entropy, pillar_offsets, counts):
        selected = torch.zeros_like(entropy, dtype=torch.bool)
        for batch_idx in range(len(pillar_offsets) - 1):
            start = int(pillar_offsets[batch_idx])
            end = int(pillar_offsets[batch_idx + 1])
            occupied = int(counts[start:end].gt(0).sum())
            if occupied == 0:
                continue
            if self.extra_ratio <= 0:
                continue
            budget = max(1, int(math.ceil(occupied * self.extra_ratio)))
            scores = entropy[start : start + occupied]
            budget = min(budget, int(scores.ge(self.min_entropy).sum()))
            if budget > 0:
                indices = scores.topk(budget, sorted=True).indices + start
                selected[indices] = True
        return selected

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
        channels = point_features.shape[1]
        z = point_xyz[:, 2].float()

        counts, z_mean, _, entropy, normalized_z = self._vertical_statistics(
            z, point_to_pillar, num_pillars
        )
        selected = self._select_split_pillars(entropy, pillar_offsets, counts)

        base_features = torch.zeros(
            num_pillars,
            channels,
            dtype=point_features.dtype,
            device=point_features.device,
        )
        expanded_idx = point_to_pillar[:, None].expand(-1, channels)
        base_features.scatter_reduce_(
            0, expanded_idx, point_features, reduce="amax", include_self=False
        )

        # A max-pooled feature is assembled from several points, so its height
        # should use a feature-weighted representative rather than an arbitrary
        # point or the raw pillar mean.
        feature_strength = point_features.float().square().mean(dim=1).sqrt()
        z_weight_sum = torch.zeros(
            num_pillars, 1, dtype=torch.float32, device=point_features.device
        )
        z_weight_sum.scatter_add_(
            0, point_to_pillar[:, None], feature_strength[:, None]
        )
        z_feature_sum = torch.zeros_like(z_weight_sum)
        z_feature_sum.scatter_add_(
            0, point_to_pillar[:, None], z[:, None] * feature_strength[:, None]
        )
        z_representative = (
            z_feature_sum[:, 0] / z_weight_sum[:, 0].clamp_min(1e-6)
        )

        assignment_input = torch.cat(
            (point_features, normalized_z[:, None].to(point_features.dtype)), 1
        )
        logits = self.assignment(assignment_input)
        logits = logits + torch.stack((-normalized_z, normalized_z), dim=1)
        if self.training and not self.deterministic_assignment:
            assignment = F.gumbel_softmax(logits.float(), tau=self.gumbel_tau, dim=1)
        else:
            assignment = logits.float().softmax(dim=1)
        assignment = assignment.to(point_features.dtype)
        active = selected[point_to_pillar, None]
        fixed_assignment = torch.zeros_like(assignment)
        fixed_assignment[:, 0] = 1
        assignment = torch.where(active, assignment, fixed_assignment)

        prototype_features = []
        prototype_z = []
        for slot in range(2):
            weight = assignment[:, slot : slot + 1]
            weight_sum = torch.zeros(
                num_pillars, 1, dtype=point_features.dtype, device=point_features.device
            )
            weight_sum.scatter_add_(0, point_to_pillar[:, None], weight)
            feature_sum = torch.zeros_like(base_features)
            feature_sum.scatter_add_(0, expanded_idx, point_features * weight)
            prototype_features.append(feature_sum / weight_sum.clamp_min(1e-6))

            z_sum = torch.zeros(
                num_pillars, 1, dtype=torch.float32, device=point_features.device
            )
            z_sum.scatter_add_(
                0, point_to_pillar[:, None], z[:, None] * weight.float()
            )
            prototype_z.append(
                z_sum[:, 0] / weight_sum[:, 0].float().clamp_min(1e-6)
            )

        if self.preserve_anchor:
            # Keep the original Matter3DToken representation available to every
            # pillar. The second prototype is an optional contextual residual,
            # never a replacement for the discriminative max-pooled anchor.
            prototype_features[0] = base_features
            prototype_z[0] = z_representative
        else:
            prototype_features[0] = torch.where(
                selected[:, None], prototype_features[0], base_features
            )
            prototype_z[0] = torch.where(selected, prototype_z[0], z_mean)

        occupied_per_sample = []
        selected_per_sample = []
        for batch_idx in range(batch_size):
            start = int(pillar_offsets[batch_idx])
            end = int(pillar_offsets[batch_idx + 1])
            occupied = int(counts[start:end].gt(0).sum())
            occupied_per_sample.append(occupied)
            selected_per_sample.append(
                torch.nonzero(selected[start : start + occupied], as_tuple=False).flatten()
            )

        token_lengths = [
            occupied + len(split)
            for occupied, split in zip(occupied_per_sample, selected_per_sample)
        ]
        max_tokens = max(token_lengths)
        sample_tokens, sample_coords = [], []
        global_xy = torch.zeros(
            num_pillars, 2, dtype=point_features.dtype, device=point_features.device
        )
        global_xy[occupied_pillars.long()] = pillar_xy.to(point_features.dtype)
        extra_position = torch.full(
            (num_pillars,), -1, dtype=torch.long, device=point_features.device
        )

        for batch_idx, (occupied, split) in enumerate(
            zip(occupied_per_sample, selected_per_sample)
        ):
            start = int(pillar_offsets[batch_idx])
            pillar_slice = slice(start, start + occupied)
            token = prototype_features[0][pillar_slice]
            coord = torch.cat(
                (
                    global_xy[pillar_slice],
                    prototype_z[0][pillar_slice, None].to(point_features.dtype),
                ),
                dim=1,
            )
            if len(split) > 0:
                global_split = split + start
                token = torch.cat((token, prototype_features[1][global_split]), dim=0)
                coord = torch.cat(
                    (
                        coord,
                        torch.cat(
                            (
                                global_xy[global_split],
                                prototype_z[1][global_split, None].to(
                                    point_features.dtype
                                ),
                            ),
                            dim=1,
                        ),
                    ),
                    dim=0,
                )
                extra_position[global_split] = torch.arange(
                    occupied, occupied + len(split), device=point_features.device
                )
            sample_tokens.append(token)
            sample_coords.append(coord)

        token_offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=point_features.device),
                torch.tensor(
                    token_lengths, dtype=torch.long, device=point_features.device
                ).cumsum(0),
            )
        )
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
                3,
                dtype=point_features.dtype,
                device=point_features.device,
            )
            for batch_idx, (token, coord) in enumerate(zip(sample_tokens, sample_coords)):
                tokens[batch_idx, : len(token)] = token
                coords[batch_idx, : len(coord)] = coord

        point_batch = torch.bucketize(
            point_to_pillar, pillar_offsets[1:], right=True
        )
        local_pillar = point_to_pillar - pillar_offsets[point_batch]
        if pack_tokens:
            base_route = token_offsets[point_batch] + local_pillar
        else:
            base_route = point_batch * max_tokens + local_pillar
        extra_local = extra_position[point_to_pillar]
        if pack_tokens:
            extra_route = token_offsets[point_batch] + extra_local.clamp_min(0)
        else:
            extra_route = point_batch * max_tokens + extra_local.clamp_min(0)
        has_extra = extra_local.ge(0)
        extra_route = torch.where(has_extra, extra_route, base_route)
        gate = (
            self.extra_gate.sigmoid().to(assignment.dtype)
            if self.extra_gate is not None
            else assignment.new_tensor(1.0)
        )
        gated_assignment = torch.stack(
            (1.0 - gate * assignment[:, 1], gate * assignment[:, 1]), dim=1
        )
        route_weights = torch.where(
            has_extra[:, None], gated_assignment, fixed_assignment
        )

        stats = {
            "tokens": sum(token_lengths),
            "base_tokens": sum(occupied_per_sample),
            "split_tokens": sum(len(x) for x in selected_per_sample),
            "mean_entropy": entropy[counts.gt(0)].mean().detach(),
        }
        return (
            tokens,
            coords,
            base_route,
            extra_route,
            route_weights,
            stats,
            token_offsets,
        )
