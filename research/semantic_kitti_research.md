# SemanticKITTI Research Log

## Protocol

- Maximum 20 epochs per experiment.
- Same seed, data split, augmentations, optimizer, and batch size for ablations.
- Primary metric: validation mIoU. Secondary metrics: rare-class IoU and convergence speed.
- Every experiment gets an isolated timestamped log directory.

## Baseline observation

At epoch 13, the reference run reached 66.48 validation mIoU, with a best value of
67.72 at epoch 10. The weakest classes were motorcyclist (0.00), other-ground
(1.92), other-vehicle (48.92), and traffic-sign (51.16). Training mIoU continued
to rise while validation mIoU plateaued, indicating both class imbalance and
overfitting.

## Experiment 1: Normalized Pillar Occupancy Encoding

Hypothesis: BEV max pooling discards pillar point count, although occupancy is a
useful geometric cue for separating sparse objects, surfaces, and boundaries.
Injecting normalized log occupancy into each BEV token should improve validation
mIoU, especially on sparse classes, without materially increasing compute.

Method: count points per pillar, apply `log1p`, normalize within each scan, and
multiply the scalar by a learned token embedding. The embedding is initialized to
zero, making step 0 exactly equivalent to the baseline while allowing occupancy
information to emerge through training.

Configurations:

- `configs/semantic_kitti/Matter3DToken_B-baseline-20.yaml`
- `configs/semantic_kitti/Matter3DToken_B-density-20.yaml`

Decision rule: retain the method if it improves best validation mIoU by at least
0.3 points or improves the mean IoU of the four weakest baseline classes by at
least 1.0 point without reducing overall mIoU.

### Run notes

- The baseline completed 20 epochs with 67.87 best validation mIoU.
- The first density run failed on its first batch because multiplying a bfloat16
  density tensor by a float32 parameter promoted the BEV tensor to float32. The
  subsequent padding operation correctly rejected mixed scalar types. The fix
  explicitly casts the learned density embedding to the point-feature dtype.

## Experiment 2: Z-aware Hierarchical Prototype Tokenizer

Each scan keeps one base token per occupied pillar and allocates a second token to
the 8% of pillars with highest normalized vertical entropy. Selected pillars use
a learned Gumbel-Softmax point-to-prototype assignment initialized with relative
height. Prototype XYZ centroids drive 3D RoPE, and the same soft assignment lifts
the two transformed prototype features back to each point. The 8% budget keeps
the original global batch size while bounding attention memory growth.

## Experiment 3: Budgeted Adaptive Pillars

The v2 tokenizer retains 85% of occupied fine-pillar tokens and assigns the
remaining regions to 1 m parent tokens. Its selector combines vertical spread,
inverse density, and a learned semantic score. At validation step 8 it reached
65.82 mIoU versus 66.62 for the baseline at the same step. The largest class
regressions were fence (-8.25), other-ground (-7.72), and truck (-4.39), while
person (+2.55), bicyclist (+4.66), and pole (+3.52) improved. This pattern shows
that sparse-object allocation works, but max pooling merged children destroys
boundary and within-object detail.

### v3: Heterogeneity Budget with Detail Relay

The optimized selector adds normalized within-parent feature heterogeneity to
the split score and balances it equally with inverse density. This targets
semantic boundaries that are not identifiable from height variance alone.

For children that remain merged, a zero-initialized detail relay projects the
child-minus-parent feature residual and its relative XY offset back after the
Transformer. All children share one transformed parent context; the relay is a
decoder-side residual and does not create a second token stream or increase
attention length. It adds 592,896 parameters (about 0.69% of the 85.9M model).

On two real SemanticKITTI validation scans (137,668 points), v2 and v3 both
produced 19,084 tokens from 22,453 occupied fine pillars (84.9953%). The v3
selector and routes passed finite forward/backward tests without changing the
token budget. The v2 run remains scheduled for all 20 epochs as the no-relay
ablation; v3 will also run all 20 epochs with the same seed and training setup.

Decision rule: retain v3 if it reduces the best-mIoU gap to the baseline while
keeping the 85% token budget, and either recovers the mean IoU of fence,
other-ground, and truck or delivers a favorable measured speed-memory tradeoff.
