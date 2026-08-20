# TennisVAR: Stroke-Evidence-Grounded Tactical Reasoning for Tennis Videos

Official implementation of **TennisVAR: A Stroke-Evidence-Grounded Multimodal Large Language Model for Tactical Reasoning in Tennis Videos**.

<p align="center">
  <a href="https://whynotgit2025.github.io/TennisVAR/"><b>Project Page</b></a> ·
  <a href="https://arxiv.org/abs/2608.12920"><b>Paper</b></a> ·
  <b>Dataset (Coming Soon)</b> ·
  <b>Checkpoints (Coming Soon)</b>
</p>

<p align="center">
  <img src="docs/public/tennisvar-figure.png" width="92%" alt="TennisVAR architecture">
</p>

TennisVAR reasons through a rally using an **event → relation → evidence → tactic** pipeline. It first parses contact-centered stroke events, models temporal and same-player dependencies, selects question-relevant evidence, and finally generates a grounded tactical answer.

## News

- **[2026.08]** 🎾 Core model code and training interfaces are released.
- **[2026.08]** 📄 The TennisVAR preprint is available on [arXiv](https://arxiv.org/abs/2608.12920).
- **Coming next:** TRACE release plan and pretrained checkpoints.

## Highlights

- **TRACE:** 11,189 rally videos with stroke events, tactical units, grounded questions, and ordered evidence chains.
- **EPM:** converts continuous rallies into explicit stroke-event sequences anchored at racket-ball contact frames.
- **TGTR:** models rally progression and same-player decision dependencies with an Evidence Router.
- **Grounded generation:** Qwen3-VL answers from sparse global frames, local evidence windows, and predicted events.

## Repository

```text
configs/                 Paper-aligned configuration
scripts/                 TGTR and Qwen3-VL training entry points
src/tennisvar/           Core EPM, TGTR and generation code
tests/                    Lightweight unit tests
docs/                     Project website
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- CUDA is required for Qwen3-VL LoRA training

```bash
git clone https://github.com/WhynotGit2025/TennisVAR.git
cd TennisVAR
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,generation,video]'
cp configs/paths.example.yaml configs/paths.local.yaml
```

Update `configs/paths.local.yaml` with your local DINOv3, Qwen3-VL, data, and artifact paths.

## Inference

The public interface is ready, while pretrained checkpoints are still being prepared for release.

```bash
tennisvar predict \
  --video /path/to/rally.mp4 \
  --ball-track /path/to/trajectory.json \
  --event-checkpoint /path/to/epm.pt \
  --tgtr-checkpoint /path/to/tgtr.pt \
  --qwen-model /path/to/Qwen3-VL-8B-Instruct \
  --question "How did the player create the winning opportunity?" \
  --output result.json
```

## Training

The checked-in defaults match the paper: EPM is trained for 40 epochs, TGTR for 120 epochs, and Qwen3-VL-8B uses rank-32 LoRA for 5 epochs.

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

## Data and Checkpoints

TRACE annotations and pretrained weights are not included yet. Professional tennis broadcasts may be controlled by third-party rights holders, so we are evaluating a research-friendly release that does not redistribute protected footage. Release updates will be posted here.

## Citation

```bibtex
@article{mei2026tennisvar,
  title   = {TennisVAR: A Stroke-Evidence-Grounded Multimodal Large Language Model for Tactical Reasoning in Tennis Videos},
  author  = {Mei, Yifan and Shi, Qingling and Wu, Changli and Rao, Jiayuan and Ji, Jiayi and Cao, Liujuan},
  journal = {arXiv preprint arXiv:2608.12920},
  year    = {2026}
}
```

## TODO

- [x] Release the paper and project page
- [x] Release the core implementation
- [ ] Release pretrained checkpoints
- [ ] Announce the TRACE access procedure
