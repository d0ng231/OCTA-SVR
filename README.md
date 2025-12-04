# OCTA-SVR (Synthetic Vasculature Reasoning)

Public release of the core code used in the **Synthetic Vasculature Reasoning (SVR)** paper: pathology-aware OCTA synthesis, automatic reasoning text generation, and minimal LLaMA-Factory configs for training/inference.

- Pathology-aware synthesis and VLM pair creation are under `pathology/` (compatible with [OCTA-autosegmentation](https://github.com/aiforvision/OCTA-autosegmentation)).
- LLaMA-Factory assets are under `llama_factory/` (upstream repo: https://github.com/hiyouga/LLaMA-Factory).
- **OCTA-100K-SVR dataset**: https://zenodo.org/records/17706653

## Quickstart
```bash
git clone https://github.com/d0ng231/OCTA-SVR
cd OCTA-SVR
python3 -m venv .venv  # Python >= 3.10 recommended
source .venv/bin/activate
pip install -r requirements.txt
```

## Synthetic pathology generation
### 1) Standard pathology-aware dataset (dropout + MA + NV)

To generate a pathology-aware OCTA-style dataset with capillary dropout, microaneurysms, neovascular tufts and mild tortuosity, use the provided config:

```bash
# Graphs + vessel maps + overlays (no GAN, fast)
python pathology/generate_vlm_dataset.py \
  --config_file pathology/generation_vlm_dropout_ma.yml \
  --stage graphs

# Full pipeline with GAN + image–text pairs
python pathology/generate_vlm_dataset.py \
  --config_file pathology/generation_vlm_dropout_ma.yml \
  --stage all \
  --use_gan \
  --gan_config /path/to/OCTA-autosegmentation/docker/trained_models/GAN/config.yml \
  --gan_epoch 150 \
  --gan_device cuda
```

Outputs (under `data/vlm_dataset_dropout_ma/` by default):

- `vessel_graphs/**/`: CSV graphs, `art_ven_img_gray.png`, `pathology_overlay_white.png`, `pathology_vis.png`, `dropout_ma.json`, `pathology.yml`.
- `images/M_*.png`: panned/cropped vessel maps with pathology overlays (optionally GAN-based OCTA if `--use_gan` is enabled).
- `metadata.jsonl`, `pairs.jsonl`: reasoning-ready VLM data (ShareGPT-style JSONL).

The YAML `pathology/generation_vlm_dropout_ma.yml` is a simplified port of
`configs/generation_vlm_dropout_ma.yml` from [OCTA-autosegmentation](https://github.com/aiforvision/OCTA-autosegmentation), keeping only the knobs needed for SVR.

### 2) Low-level vessel graph + vessel map demo (no pathology)

```bash
python pathology/generate_synthetic_octa_images.py \
  --num_samples 100 \
  --out_dir data/vlm_dataset_demo \
  --workers -1
```
This creates per-sample graph folders under `data/vlm_dataset_demo/vessel_graphs/` that contain CSV graphs and grayscale vessel maps (`art_ven_img_gray.png`).

To additionally render realistic OCTA images with the GAN from [OCTA-autosegmentation](https://github.com/aiforvision/OCTA-autosegmentation), append:
```bash
  --use_gan \
  --gan_config /path/to/gan/config.yml \
  --gan_epoch 150
```
GAN configs/weights (and the `test.py` runner) live in the upstream OCTA-autosegmentation repo; to use GAN rendering, run this script inside that repo or copy it into that workspace. In most SVR use cases you do **not** need to call this script directly—prefer the config-driven `generate_vlm_dataset.py` pipeline above.

### 3) Generate VLM text pairs

If you already have graphs/images under a dataset root, you can regenerate only the VLM text pairs (without re-running growth or GAN) via:

```bash
python pathology/generate_vlm_dataset.py \
  --config_file pathology/generation_vlm_dropout_ma.yml \
  --out_dir data/vlm_dataset_dropout_ma \
  --stage pairs
```

This reads existing `vessel_graphs/` and `images/` under `data/vlm_dataset_dropout_ma/` and writes/updates `metadata.jsonl` and `pairs.jsonl` (ShareGPT-style conversations) in that folder.

Key outputs for any dataset root: `images/*.png`, `metadata.jsonl`, and `pairs.jsonl`. Pathology ranges and profiles are best adjusted via the YAML configs in `pathology/` rather than editing Python.

## Training with LLaMA-Factory
Place `OCTA-100K-SVR` under `data/OCTA-100K-SVR/` (so `pairs_diversified.jsonl` and `images/` live there), then run:
```bash
bash llama_factory/scripts/run_svr_training.sh
```
The script calls `llamafactory-cli` with `llama_factory/configs/svr_qwen3vl_full.yaml` and `llama_factory/data/dataset_info.json`. Edit the config to swap the base model or adjust batch size/epochs. Refer to the upstream LLaMA-Factory docs for environment/setup details.

## Inference
```bash
# Single image
bash llama_factory/scripts/run_inference.sh \
  MODEL_PATH=/path/to/qwen3vl_svr_checkpoint \
  IMAGE_PATH=/path/to/image.png \
  OUT_PATH=outputs/predictions.jsonl
```
The underlying helper `llama_factory/inference_octa_CoT.py` supports both single-image and folder inference, optional LoRA adapters, and template selection (`qwen3_vl`, `qwen2_vl`, etc.).

## Repo layout
- `pathology/`: Synthetic vessel/pathology simulation, CoT pairing, cleanup.
- `pathology/vessel_graph_generation/`: Graph-growth simulator and rasterization.
- `llama_factory/`: Training config, dataset descriptor, inference helpers, scripts.
- `requirements.txt`: Minimal deps (GAN weights/models not included).
