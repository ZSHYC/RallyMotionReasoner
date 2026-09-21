# TennisVAR algorithm update

This document records the algorithm changes made after the original TennisVAR
implementation and the preceding `724b5e3` update. It describes the intended
model structure only; no training run or numerical reproduction is claimed.

## Current pipeline

```text
EPM event features
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

## TGTR-v2 changes

### Stroke representation

The tokenizer now receives the six available structured values for each stroke:

`ball_x`, `ball_y`, `ball_visible`, `ball_mask`, `contact_frame`, and
`contact_mask`.

They are projected once and fused with the existing 800-dimensional EPM event
feature. Optional cached motion statistics remain supported through the existing
`motion_token_mode`/`motion_feature_dim` settings. Missing values are represented
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

## What the preceding update fixed

The previous update introduced optimal one-to-one temporal matching, explicit
null/invisible-ball handling, frame-rate-aware deltas, EPM masking and
sinusoidal token positions, CJK tokenization, removal of train-only QA metadata,
evidence/key subset constraints, unknown-hitter graph handling, degree-normalized
messages, padding-safe TGTR tokens, and structured pipeline fields. These are
kept in TGTR-v2 and form its data contract.

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

TGTR-v2 changes parameter shapes in the graph and stroke encoders. Existing
TGTR checkpoints therefore need to be retrained with the updated structure; no
compatibility or migration layer is added. This repository update intentionally
does not run training or report a new numerical result. The lightweight tests
only check tensor contracts, masking, finite outputs, and the evidence subset
interface.
