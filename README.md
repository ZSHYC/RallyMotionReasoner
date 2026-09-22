# RallyMotionReasoner

RallyMotionReasoner is a modular system for event-grounded tactical reasoning in tennis videos. It converts a rally into temporally aligned hit and bounce events, builds a tactical graph, retrieves evidence for a question, and optionally generates a grounded answer with a Qwen vision-language model.

The repository contains the model structure and integration code. It does not claim reproduced training results and does not include broadcast videos or pretrained weights.

## What the system does

```text
video + ball trajectory
        │
        ▼
region-motion event detector
        │  hit / bounce / timing / motion statistics
        ▼
tactical event graph
        │  temporal edges + same-player edges + bounce cues
        ▼
TGTR temporal graph reasoner
        │  shot tokens + motion tokens + graph relations
        ▼
evidence router
        │  question-conditioned evidence and key actions
        ▼
optional Qwen vision-language generator
```

### Event detection

The event detector is the only event backend in this repository. It combines two expert inputs:

- **TrajectoryExpert** encodes ball position, velocity, acceleration, visibility, and temporal validity from TrackNet-style trajectories.
- **VisualExpert** encodes a full-frame DINO feature together with four spatial crops, preserving both court context and local player motion.
- **Bidirectional cross-attention** lets trajectory tokens query visual regions and visual regions query trajectory context.
- **Event heads** predict event confidence, event type, hitter, and optional technique attributes.
- **Masks** remove padded trajectory slots and missing visual regions before attention and pooling.

The runtime loads `trajectory_expert.pt` and `visual_expert.pt` from one expert directory. It decodes hit and bounce events with temporal suppression. Bounce events remain separate graph cues; they are not converted into player stroke nodes.

### Graph and TGTR

Hit events become stroke nodes. Bounce events provide timing context such as the gap before or after a stroke. TGTR receives:

- 800-dimensional frame descriptors;
- 24-dimensional motion statistics derived from those descriptors;
- shot timing and event attributes;
- temporal relations between neighboring strokes;
- same-player relations where the hitter is known;
- bounce-aware structural tokens.

The graph loader and event exporter use the same structured feature payload, so offline TGTR data and online inference share the frame alignment contract.

### Evidence-grounded generation

The evidence router scores candidate strokes and key actions for a tactical question. Key-action probability is computed from the conditional key score and evidence score. The generator receives validated candidate identifiers, their evidence frames, sparse global context, and predicted event information. It must return the canonical answer fields defined in [`src/tennisvar/schema.py`](src/tennisvar/schema.py).

## Requirements

- Python 3.10 or newer
- PyTorch 2.1 or newer
- OpenCV and DINOv3 dependencies for video event extraction
- CUDA is recommended for model inference and required by most large-model training setups

Install the package and optional training dependencies:

```bash
git clone https://github.com/ZSHYC/RallyMotionReasoner.git
cd RallyMotionReasoner
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,generation,video]'
```

Copy the path template and fill in local data and model locations when using the data preparation or training scripts:

```bash
cp configs/paths.example.yaml configs/paths.local.yaml
```

## Inputs

A ball trajectory can be a TrackNet JSON payload or a CSV file. The trajectory loader normalizes both formats into the same temporal representation. The event expert directory must contain:

```text
region-experts/
├── trajectory_expert.pt
└── visual_expert.pt
```

The video, trajectory, DINOv3 repository/weights, TGTR checkpoint, and optional Qwen model are supplied independently. No hash, fingerprint, or artifact digest is required.

## Commands

Show all available commands:

```bash
PYTHONPATH=src python -m tennisvar.cli --help
```

### Prepare graph and QA data

```bash
PYTHONPATH=src python -m tennisvar.cli prepare-data \
  --config configs/tennisvar.yaml
```

This validates configured splits and builds graph/QA artifacts. It does not train a model.

### Run event detection

```bash
rallymotionreasoner predict-events \
  --video /path/to/rally.mp4 \
  --ball-track /path/to/trajectory.json \
  --checkpoint /path/to/region-experts \
  --dinov3-repo /path/to/dinov3 \
  --dinov3-weights /path/to/dinov3.pth \
  --output outputs/events.json
```

Use `PYTHONPATH=src python -m tennisvar.cli predict-events` when the editable package is not installed. The output contains decoded events, frame scores, frame features, and feature provenance.

### Export events for TGTR

```bash
rallymotionreasoner export-events \
  --checkpoint /path/to/region-experts \
  --split train \
  --experiment-config configs/tennisvar.yaml \
  --dinov3-repo /path/to/dinov3 \
  --dinov3-weights /path/to/dinov3.pth \
  --output outputs/events_train
```

The exporter reads configured rally videos and TrackNet files, writes predicted event graphs, and stores hit-only frame features using the shared TGTR payload format. Run it separately for `train`, `val`, and `test` when those splits are configured.

### Run the complete pipeline

```bash
rallymotionreasoner predict \
  --video /path/to/rally.mp4 \
  --ball-track /path/to/trajectory.json \
  --event-checkpoint /path/to/region-experts \
  --tgtr-checkpoint /path/to/tgtr.pt \
  --qwen-model /path/to/Qwen3-VL-8B-Instruct \
  --question "How did the player create the winning opportunity?" \
  --dinov3-repo /path/to/dinov3 \
  --dinov3-weights /path/to/dinov3.pth \
  --output outputs/answer.json
```

The Qwen model and adapter are optional at the code level; omit them when only structured event and TGTR outputs are needed.

### Train downstream components

Training is separate from event inference and is not required for reading the model structure:

```bash
PYTHONPATH=src python scripts/train_tgtr.py \
  --experiment-config configs/tennisvar.yaml

PYTHONPATH=src torchrun --nproc-per-node=8 scripts/train_qwen_lora.py \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --train-data /path/to/train.jsonl \
  --val-data /path/to/val.jsonl \
  --output-dir runs/qwen_lora \
  --report outputs/qwen_lora.json
```

These commands describe available interfaces only. No training or numerical reproduction is part of this repository update.

## Repository layout

```text
configs/                         YAML experiment and path configuration
scripts/train_tgtr.py            TGTR training entry point
scripts/train_qwen_lora.py       Qwen LoRA training entry point
src/tennisvar/event_parsing/     event model, features, graph, decoder, runtime
src/tennisvar/features/           trajectory input and normalization
src/tennisvar/tactical_reasoning/ TGTR, motion adapter, evidence routing
src/tennisvar/generation/        structured generation and manifest handling
src/tennisvar/pipeline.py        end-to-end event → TGTR → generation flow
tests/                           lightweight contract and shape tests
docs/algorithm.md                algorithm and interface notes
```

## Validation

Run the lightweight checks used for code changes:

```bash
ruff check src scripts tests
PYTHONPATH=src pytest -q
```

The tests validate tensor shapes, masked fusion behavior, event graph semantics, feature alignment, TGTR contracts, configuration validation, and structured output fields. They do not run training or real video inference.

## License

See [`LICENSE`](LICENSE) for the applicable license and attribution requirements.
