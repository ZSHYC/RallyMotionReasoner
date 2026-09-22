# RallyMotionReasoner algorithm

RallyMotionReasoner turns a tennis rally into explicit events, graph relations, and question-conditioned evidence. The implementation is organized as one inference path:

```text
trajectory + video
  -> Region Motion Event Model
  -> hit/bounce event graph
  -> TGTR graph reasoning
  -> evidence routing
  -> optional grounded generation
```

## 1. Region Motion Event Model

The event detector combines two expert streams.

### TrajectoryExpert

`src/tennisvar/features/ball_trajectory.py` accepts TrackNet JSON or CSV and normalizes it into fixed temporal rows. Each row contains ball position, motion derivatives, visibility, and a validity indicator. The trajectory encoder models short-term movement and preserves the distinction between a valid temporal slot and a visible ball observation.

### VisualExpert

`src/tennisvar/event_parsing/features.py` extracts DINOv3 features for the full frame and four corner crops. The visual expert encodes global court context and local player regions without adding a separate event model.

### Cross-stream fusion

`src/tennisvar/event_parsing/model.py` uses bidirectional cross-attention: trajectory tokens attend to visual regions, while visual regions attend to trajectory context. Invalid trajectory slots and missing visual regions are masked before attention and pooling. Event heads predict eventness, event type, hitter, and optional technique attributes.

The runtime in `src/tennisvar/event_parsing/runtime.py` loads `trajectory_expert.pt` and `visual_expert.pt`, evaluates dense windows, fuses the expert scores, and decodes hit and bounce events. Bounce is a graph cue, not a player stroke.

## 2. Event graph and feature contract

`src/tennisvar/event_parsing/graph.py` creates stroke nodes from hit events and stores bounce events separately. Temporal edges connect neighboring strokes. Same-player edges are added only when the hitter is known. Stroke nodes retain event timing, confidence, attributes, and bounce gaps.

`shot_feature_payload` in `src/tennisvar/event_parsing/features.py` is the shared export contract for offline data and online inference. It aligns hit frames with 800-dimensional frame descriptors and exposes a 24-dimensional motion slice for downstream reasoning.

## 3. TGTR

TGTR consumes the event graph, shot descriptors, motion statistics, and structural tokens. It models:

- temporal progression across strokes;
- same-player tactical relations;
- bounce-aware timing;
- padding and missing observations through masks;
- frame and motion evidence alongside graph tokens.

The implementation is in `src/tennisvar/tactical_reasoning/`. Event data is generated with `event_backend=region_fusion`; no alternate event backend is maintained.

## 4. Evidence routing and generation

The evidence router ranks candidate strokes and key actions using the question representation and TGTR state. Key-action scores combine evidence and conditional key probabilities. The generator receives validated candidate IDs, selected evidence frames, global context, and predicted event information. Output validation is defined in `src/tennisvar/schema.py`.

## 5. Data and checkpoints

The repository keeps checkpoint and data validation structural: required files, dimensions, label maps, vocabulary, and schemas are checked directly. It does not compute artifact hashes, fingerprints, or checksums.

External artifacts are intentionally separate:

```text
region-experts/
├── trajectory_expert.pt
└── visual_expert.pt
```

TGTR and optional Qwen adapters are loaded by their own stages. This separation keeps event extraction, graph reasoning, and generation independently inspectable.

## 6. Code map

| Area | Location | Responsibility |
| --- | --- | --- |
| Trajectory input | `src/tennisvar/features/ball_trajectory.py` | JSON/CSV loading and normalization |
| Event model | `src/tennisvar/event_parsing/model.py` | trajectory/visual experts and fusion |
| Event features | `src/tennisvar/event_parsing/features.py` | DINO features and TGTR payloads |
| Event runtime | `src/tennisvar/event_parsing/runtime.py` | expert loading and decoding |
| Event graph | `src/tennisvar/event_parsing/graph.py` | hit/bounce nodes and relations |
| TGTR | `src/tennisvar/tactical_reasoning/` | graph-temporal reasoning |
| Generation | `src/tennisvar/generation/` | structured grounded answers |

## 7. Verification scope

The lightweight test suite checks tensor shapes, masking, event graph semantics, frame alignment, TGTR contracts, configuration validation, and structured outputs. It does not train models, process real videos, or report reproduced numerical results.
