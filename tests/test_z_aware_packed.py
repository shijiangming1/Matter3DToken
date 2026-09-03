#!/usr/bin/env python3
"""Check padded and packed Z-prototype routing on two variable-length scans."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matter3dtoken.z_aware_prototype_tokenizer import ZAwarePrototypeTokenizer
from matter3dtoken.transformer import Transformer


def test_tokenizer_routes():
    torch.manual_seed(3)
    tokenizer = ZAwarePrototypeTokenizer(
        4,
        extra_ratio=0.5,
        min_entropy=0.0,
        preserve_anchor=True,
        deterministic_assignment=True,
    )
    features = torch.randn(10, 4)
    xyz = torch.tensor(
        [[0.0, 0.0, height] for height in (-1, 1, 0, 2, -2, 2, -1, 1, -0.5, 0.5)]
    )
    point_to_pillar = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    pillar_xy = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    occupied = torch.arange(5)
    lengths = torch.tensor([3, 2])
    pad_token = torch.zeros(4)

    packed = tokenizer(
        features, xyz, point_to_pillar, pillar_xy, occupied, lengths, 2, pad_token, True
    )
    padded = tokenizer(
        features, xyz, point_to_pillar, pillar_xy, occupied, lengths, 2, pad_token, False
    )
    ptok, pcoord, pbase, pextra, pweight, pstats, poff = packed
    btok, bcoord, bbase, bextra, bweight, bstats, boff = padded
    assert pstats == bstats
    assert torch.equal(pweight, bweight)
    assert torch.equal(poff, boff)
    for batch_idx in range(2):
        count = int(poff[batch_idx + 1] - poff[batch_idx])
        assert torch.allclose(ptok[poff[batch_idx] : poff[batch_idx + 1]], btok[batch_idx, :count])
        assert torch.allclose(pcoord[poff[batch_idx] : poff[batch_idx + 1]], bcoord[batch_idx, :count])

    packed_lift = ptok[pbase] * pweight[:, :1] + ptok[pextra] * pweight[:, 1:]
    padded_flat = btok.reshape(-1, 4)
    padded_lift = (
        padded_flat[bbase] * bweight[:, :1] + padded_flat[bextra] * bweight[:, 1:]
    )
    assert torch.allclose(packed_lift, padded_lift)
    assert pbase[6] >= poff[1]


def test_packed_transformer():
    model = Transformer(dim=24, depth=1, num_heads=3, expansion=1, drop_path=0).eval()
    output = model(torch.randn(5, 24), torch.randn(5, 3), torch.tensor([0, 3, 5]))
    assert output.shape == (5, 24)
    assert torch.isfinite(output).all()


if __name__ == "__main__":
    test_tokenizer_routes()
    test_packed_transformer()
    print("packed Z-aware Prototype Tokenizer checks passed")
