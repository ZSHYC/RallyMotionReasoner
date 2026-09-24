# RallyMotionReasoner algorithm

RallyMotionReasoner turns a tennis rally into explicit events, graph relations, and question-conditioned evidence. The implementation is organized as one inference path:

```text
trajectory + video
  -> Region Motion Event Model
  -> hit/bounce event graph
  -> RGR graph reasoning
  -> evidence routing
  -> optional grounded generation
```

## 1. Region Motion Event Model

The event detector combines two expert streams.

### TrajectoryExpert

`src/rallymotionreasoner/features/ball_trajectory.py` accepts TrackNet JSON or CSV and normalizes it into fixed temporal rows. Each row contains ball position, motion derivatives, visibility, and a validity indicator. The trajectory encoder models short-term movement and preserves the distinction between a valid temporal slot and a visible ball observation. Its released expert weights retain a 779-column projection, with the first 11 columns used for the 11 trajectory inputs.

### VisualExpert

`src/rallymotionreasoner/event_detection/features.py` extracts DINOv3 features for the full frame and four corner crops. The visual expert encodes global court context and local player regions without adding a separate event model.

### Event score fusion

`src/rallymotionreasoner/event_detection/model.py` defines a joint `MotionRegionEventModel` with bidirectional cross-attention and optional attribute heads. It has no checkpointed training or inference path in this repository. The current runtime evaluates `TrajectoryExpert` and `VisualExpert` independently and averages their eventness/type scores. Its decoded hitter and technique attributes remain unknown.

The runtime in `src/rallymotionreasoner/event_detection/runtime.py` loads `trajectory_expert.pt` and `visual_expert.pt`, evaluates dense windows, fuses the expert scores, and decodes hit and bounce events. Bounce is a graph cue, not a player stroke.

## 2. Event graph and feature contract

`src/rallymotionreasoner/event_detection/graph.py` creates stroke nodes from hit events and stores bounce events separately. Temporal edges connect neighboring strokes. Same-player edges require known hitter labels, which the current event runtime does not produce. Stroke nodes retain event timing, confidence, available attributes, and bounce gaps.

`shot_feature_payload` in `src/rallymotionreasoner/event_detection/features.py` is the shared export contract for offline data and online inference. It aligns hit frames with 800-dimensional frame descriptors and exposes a 24-dimensional motion slice for downstream reasoning.

## 3. RGR

RGR consumes the event graph, shot descriptors, motion statistics, and structural tokens. It models:

- temporal progression across strokes;
- same-player tactical relations;
- bounce-aware timing;
- padding and missing observations through masks;
- frame and motion evidence alongside graph tokens.

The implementation is in `src/rallymotionreasoner/graph_reasoning/`. Event data is generated with `event_backend=motion_region`; no alternate event backend is maintained.

Ball-position and contact-time heads are optional auxiliary targets. They receive supervision only when `feature_extraction.visual_supervision_path` is configured; the default configuration does not provide that file. These target values are not fed into the RGR encoder.

RGR uses the checkpoint's `max_strokes` limit in training and inference. Evidence-frame tolerance is calibrated at 25 FPS and scaled to each graph's FPS; key-action targets are restricted to evidence targets. The default `time_bias_version: 2` gives zero temporal distance its own attention bucket. Checkpoints without that field use version 1 timing.

## 4. Evidence routing and generation

The evidence router ranks candidate strokes and key actions using the question representation and RGR state. Key-action scores combine evidence and conditional key probabilities. Candidates below the RGR evidence threshold are withheld from generation; when none qualify, the pipeline returns an unanswerable result. Otherwise, the generator receives validated candidate IDs, selected evidence frames, global context, and predicted event information. The frame budget keeps selected evidence centers, and Qwen receives their original frame indices and source FPS for timestamps. SFT rows need both `e2e_metadata.video_frame_indices` and `video_fps` to use source timing; older rows retain the utility's default timing. Output validation is defined in `src/rallymotionreasoner/schema.py`.

## 5. Data and checkpoints

Checkpoint and data validation checks required files, dimensions, label maps, vocabulary, and schemas.

External artifacts are intentionally separate:

```text
region-experts/
├── trajectory_expert.pt
└── visual_expert.pt
```

RGR and optional Qwen adapters are loaded by their own stages. This separation keeps event extraction, graph reasoning, and generation independently inspectable.

## 6. Code map

| Area | Location | Responsibility |
| --- | --- | --- |
| Trajectory input | `src/rallymotionreasoner/features/ball_trajectory.py` | JSON/CSV loading and normalization |
| Event model | `src/rallymotionreasoner/event_detection/model.py` | trajectory/visual experts and fusion |
| Event features | `src/rallymotionreasoner/event_detection/features.py` | DINO features and RGR payloads |
| Event runtime | `src/rallymotionreasoner/event_detection/runtime.py` | expert loading and decoding |
| Event graph | `src/rallymotionreasoner/event_detection/graph.py` | hit/bounce nodes and relations |
| RGR | `src/rallymotionreasoner/graph_reasoning/` | graph-temporal reasoning |
| Generation | `src/rallymotionreasoner/generation/` | structured grounded answers |

## 7. Verification scope

The lightweight test suite checks tensor shapes, masking, event graph semantics, frame alignment, RGR contracts, configuration validation, and structured outputs. It does not train models, process real videos, or report reproduced numerical results.
