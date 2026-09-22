# TennisVAR algorithm update

This document records the algorithm changes made after the original TennisVAR
implementation and the preceding `724b5e3` update. It describes the intended
model structure only; no training run or numerical reproduction is claimed.

## Current pipeline

```text
Region motion event features
  -> structured stroke state (ball/contact masks and normalized times)
  -> semantic + visual stroke tokenizer
  -> local multiscale temporal mixer (depthwise 3/5-frame paths)
  -> relation-temporal graph Transformer
  -> question-conditioned evidence/key-action router
  -> label heads and Qwen3-VL structured generation
```

The graph still has the two task-defined directed relations
`temporal_next` and `same_player_next`. The update increases how those
relations are used rather than inventing speculative edge types.

## Region event detector

The event detector accepts the `tennis-region-infer` expert directory as its
only event artifact. A directory containing `trajectory_expert.pt` and
`visual_expert.pt` selects the trajectory and five-view region experts directly.

The trajectory branch represents each frame with normalized position,
visibility, velocity, acceleration, turning, curvature, validity, and the
sampled time offset (11 values). It reads a 0.4-second, 25-sample window. The
visual branch encodes one full frame and four 55%-area corner crops with the
shared DINOv3 CLS encoder, producing 3840 values per frame over a 1.6-second,
49-sample window. Each branch predicts eventness and event type; their product
scores are averaged 1:1 and decoded with per-class radius-5 NMS.

`bounce` is retained as a physical event cue but is not turned into a stroke
node. Predicted `hit` events remain graph stroke nodes, while bounce frames are
stored beside the graph for temporal supervision and later contact-aware
extensions. This prevents a ball-ground interaction from being mistaken for a
player action or corrupting same-player relations.

For region graphs, the nearest preceding and following bounce gaps are also
encoded as structural stroke tokens. TGTR can use bounce timing as context
without adding bounce nodes or changing player-relation semantics.

`RegionFusionEventModel` adds bidirectional motion-to-region and
region-to-motion cross-attention plus the attribute heads. It is the joint
training structure; the runtime expert directory preserves the published
independent expert weights and fuses their event probabilities at inference.

The integration is based on
[`tennis-region-infer`](https://github.com/ZSHYC/tennis-region-infer),
Trajectory Attention/Motionformer ([arXiv:2106.05392](https://arxiv.org/abs/2106.05392)),
and T-DEED ([arXiv:2404.05392](https://arxiv.org/abs/2404.05392)). These
references motivate the representation and temporal design; this repository
does not run training, inference reproduction, or report a new metric.

## TGTR changes

### Stroke representation

The tokenizer now receives the six available structured values for each stroke:

`ball_x`, `ball_y`, `ball_visible`, `ball_mask`, `contact_frame`, and
`contact_mask`.

They are projected once and fused with the existing 800-dimensional event
feature. Optional cached motion statistics remain supported through the existing
`motion_feature_dim` setting (0 or 24). Missing values are represented
by their masks and zero-filled coordinates, so the model does not confuse an
unobserved ball with an observed coordinate at the origin.

The local mixer applies depthwise temporal convolutions with kernel sizes 3 and
5, then a gated residual fusion. This keeps contact-local detail while giving a
stroke access to short tactical context before graph reasoning.

### Relation-temporal graph block

For a directed edge (j\rightarrow i) with relation (r_{ij}), the block first
adds a degree-normalized message:

\[
m_i = \sum_{j\rightarrow i}
\frac{\operatorname{MLP}([h_j,e_{r_{ij}}])}{\sqrt{\max(1,d_i)}}.
\]

The resulting tokens use multi-head self-attention. Each attention logit gets a
learned relation bias and a signed bucketed time bias:

\[
\operatorname{logit}_{ij}^{(k)} \leftarrow
\operatorname{logit}_{ij}^{(k)} + b_{r_{ij}}^{(k)} + t_{q(\Delta f_{ij})}^{(k)}.
\]

`node_frames` are normalized contact frames. The signed buckets distinguish
past, current, and future context while clipping large gaps. Padding keys are
masked and padded query outputs are zeroed after every block. Three blocks are
the default configuration in `configs/tennisvar.yaml`; the constructor remains
parameterized for other research settings.

Graph pooling is learned rather than an unweighted mean, allowing the model to
preserve the most decision-relevant stroke before evidence routing.

### Evidence routing

Evidence remains a multi-label prediction. The router now converts each
evidence logit with an independent sigmoid gate and normalizes only for the
context mixture. This avoids making one stroke win a softmax competition when a
tactical explanation needs several supporting strokes. Key-action logits remain
conditioned by the evidence log-probability, so a decisive action stays inside
the predicted evidence chain.

The reasoner also exposes auxiliary per-stroke heads for normalized ball
position, ball visibility, and contact frame. When visual supervision is
available, the existing masks determine which targets contribute to the loss;
without that supervision these terms remain inactive. Evidence and key-action
BCE terms use a capped batch-local positive weight, which prevents sparse event
labels from being optimized as an all-negative problem.

Key actions are evaluated as a multi-label set. Thresholded key predictions
with an argmax fallback are compared with set F1, while exact-set accuracy is
retained as a secondary diagnostic. Model selection uses key-action F1 so its
objective matches the target representation.

## What the preceding update fixed

The previous update introduced optimal one-to-one temporal matching, explicit
null/invisible-ball handling, frame-rate-aware deltas, event masking and
sinusoidal token positions, CJK tokenization, removal of train-only QA metadata,
evidence/key subset constraints, unknown-hitter graph handling, degree-normalized
messages, padding-safe TGTR tokens, and structured pipeline fields. These are
kept in TGTR and form its data contract.

## Research basis and scope

The design is informed by the following public directions:

- [TennisVAR](https://arxiv.org/abs/2608.12920): event-relation-evidence-tactic
  decomposition used as the project baseline.
- [T-DEED](https://arxiv.org/abs/2404.05392) and precise event spotting work:
  local multiscale temporal discrimination.
- [VideoITG](https://arxiv.org/html/2507.13353v2) and
  [UniTime](https://arxiv.org/abs/2506.18883): adaptive and multiscale temporal
  grounding.
- [QGAC-TR](https://aclanthology.org/2024.findings-emnlp.176/):
  question-guided, answer-calibrated temporal localization.
- [GraphThinker](https://arxiv.org/html/2602.17555v3): explicit event relations
  as a grounding scaffold for video reasoning.
- ViTED (CVPR 2025), TimeCraft (ECCV 2024), and TimeRefine (WACV 2026):
  evidence chains and coarse-to-fine temporal boundaries. Only the lightweight
  parts compatible with the current labels are adopted here; no new annotation
  or external agent loop is assumed.
- [RacketVision](https://github.com/OrcustD/RacketVision) and
  [TOTNet](https://arxiv.org/html/2508.09650): perception-side motivation for
  preserving contact and occlusion-aware cues in the event representation.

These references motivate architectural choices; they do not imply that the
repository reproduces their datasets, checkpoints, or reported metrics.

## Checkpoint and evaluation boundary

TGTR changes parameter shapes in the graph and stroke encoders. A TGTR
checkpoint used with the region detector must declare
`event_backend=region_fusion` and be trained from graphs generated with the
same hit-node/bounce-cue contract. No migration layer or numerical result is
claimed; lightweight checks only validate tensor contracts and graph semantics.

The training entry point records `event_backend=region_fusion` and reads the
actual exported hit features. Training does not require test-split artifacts.

## Module replacement and cleanup

`event_parsing/model.py`, `features.py`, and `runtime.py` now contain the region
experts, feature extraction, and sole event predictor. Trajectory CSV/JSON input
and normalization live in `features/ball_trajectory.py`. The superseded event
model, trainer, cache decoder, and compatibility aliases have been removed.
The 800-D frame descriptor remains because TGTR consumes it; it is not an old
event detector. `export-events` uses the current predictor and exports hit-only
features through the same payload builder as live TGTR inference.

The joint cross-attention model and attribute heads remain available as model
structure. Published independent expert checkpoints do not contain those
parameters: their runtime returns unknown attributes rather than invented
predictions, so same-player edges require a predicted hitter. Cross-attention
masks exclude padded slots, including an entirely missing stream. The trajectory
`valid` channel marks real temporal slots; `detected` separately marks visibility.

Artifact digest computation and fingerprint metadata have been removed throughout
data preparation, event features, checkpoints, and generation. QA identifiers use
split, rally, and question index. Resume compares configuration and label/vocabulary
maps directly. Structural dimensions, required files, and schema checks remain.

This pass also fixes the stored TGTR threshold being ignored, candidate indexing
for rallies exceeding 32 hits, CPU best-weight snapshots sharing mutable storage,
and distributed sample padding when there are fewer samples than ranks. Key-action
probability is now the product of its conditional probability and evidence
probability. Final answer evidence follows the generator's validated candidate IDs;
TGTR predictions remain separately available in provenance.
