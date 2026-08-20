# TennisVAR

[![Project page](https://img.shields.io/badge/project-page-2f6f4e)](https://whynotgit2025.github.io/TennisVAR/)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776ab)](https://www.python.org/)
[![License](https://img.shields.io/badge/code-Apache--2.0-blue)](LICENSE)

Official core implementation of **TennisVAR: A Stroke-Evidence-Grounded Multimodal Large Language Model for Tactical Reasoning in Tennis Videos**.

TennisVAR follows an `event -> relation -> evidence -> tactic` pipeline. The Event Parsing Module (EPM) converts a rally into contact-centered stroke events. The Tactical Graph-Guided Temporal Reasoner (TGTR) connects adjacent strokes and same-player decisions, routes question-relevant evidence, and predicts hierarchical tactics. Qwen3-VL realizes the grounded answer from sparse global frames, local evidence windows, and the predicted event table.

## Release status

| Component | Status |
| --- | --- |
| Core EPM, TGTR, Evidence Router and Qwen3-VL integration | Available |
| TRACE annotations and split metadata | Release under review |
| Tennis broadcast videos | Not distributed |
| Pretrained EPM, TGTR and LoRA weights | Coming soon |
| Paper | Coming soon |

The data release is being designed separately because professional match footage may be subject to third-party rights. See [DATA_AND_MODELS.md](DATA_AND_MODELS.md) before opening a data-related issue.

## Open-source scope

This repository contains the paper-facing model code, training entry points, configuration, artifact validation, and tests. It does not contain tournament footage, extracted frames, TRACE annotations, checkpoints, cached features, experiment sweeps, or evaluation outputs.

```text
configs/tennisvar.yaml                 paper configuration
scripts/train_tgtr.py                  TGTR training entry point
scripts/train_qwen_lora.py             Qwen3-VL LoRA training entry point
src/tennisvar/event_parsing/           EPM and event decoding
src/tennisvar/tactical_reasoning/      TGTR and Evidence Router
src/tennisvar/generation/              grounded answer generation
src/tennisvar/data/                    TRACE-compatible data interfaces
docs/                                  GitHub Pages project website
```

## Installation

```bash
git clone https://github.com/WhynotGit2025/TennisVAR.git
cd TennisVAR
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,generation,video]'
cp configs/paths.example.yaml configs/paths.local.yaml
```

Edit `configs/paths.local.yaml` to reference assets that you are authorized to use. DINOv3 and Qwen3-VL remain external dependencies. TrackNet is integrated through a trajectory JSON interface; third-party TrackNet code and weights are not redistributed here.

## Paper configuration

The checked-in configuration matches the method description:

- EPM input: 768-D DINOv3 appearance + 24-D motion + 8-D ball trajectory = 800-D.
- EPM training: 40 epochs, batch size 64, AdamW learning rate `2e-4`.
- TGTR: 256-D hidden states with temporal and same-player relations.
- TGTR training: 120 epochs, batch size 64, AdamW learning rate `1.5e-3`.
- Evidence and key-action loss weights: `2.0`; three tactic-level weights: `1.0`.
- Generator: Qwen3-VL-8B with rank-32 LoRA for 5 epochs at `2e-5`, effective batch size 32.

## Commands

The commands below require the unreleased checkpoints and compatible data artifacts. They are provided now so the public interfaces remain reviewable.

```bash
tennisvar predict-events \
  --video /path/to/authorized/rally.mp4 \
  --ball-track /path/to/tracknet/trajectory.json \
  --checkpoint /path/to/epm.pt \
  --output outputs/events.json

PYTHONPATH=src python scripts/train_tgtr.py \
  --experiment-config configs/tennisvar.yaml

PYTHONPATH=src torchrun --nproc-per-node=8 scripts/train_qwen_lora.py \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --train-data /path/to/train.jsonl \
  --val-data /path/to/val.jsonl \
  --output-dir runs/qwen_lora \
  --report outputs/qwen_lora.json \
  --epochs 5 --learning-rate 2e-5 --lora-rank 32
```

## Development

```bash
python -m pip install -e '.[dev,train]'
ruff check .
pytest
```

Contributions should not include match videos, extracted frames, private annotations, credentials, local paths, or model weights. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

```bibtex
@article{mei2026tennisvar,
  title  = {TennisVAR: A Stroke-Evidence-Grounded Multimodal Large Language Model for Tactical Reasoning in Tennis Videos},
  author = {Mei, Yifan and Shi, Qingling and Wu, Changli and Rao, Jiayuan and Ji, Jiayi and Cao, Liujuan},
  year   = {2026}
}
```

## License

The source code is released under the [Apache License 2.0](LICENSE). This license does not grant rights to tennis broadcasts, TRACE data, third-party model weights, or external dependencies.
