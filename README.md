# OCTA-SVR (Synthetic Vasculature Reasoning)

Public release of the core code used in the **Synthetic Vasculature Reasoning (SVR)** paper: pathology-aware OCTA synthesis, automatic reasoning text generation, and minimal LLaMA-Factory configs for training/inference.

- Pathology-aware synthesis and VLM pair creation are under `pathology/` (compatible with [OCTA-autosegmentation](https://github.com/aiforvision/OCTA-autosegmentation)).
- LLaMA-Factory assets are under `llama_factory/` (upstream repo: https://github.com/hiyouga/LLaMA-Factory).
- **OCTA-100K-SVR dataset**: https://zenodo.org/records/17706653

## Quickstart
```bash
git clone https://github.com/d0ng231/OCTA-SVR
cd OCTA-SVR
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Synthetic pathology generation
1) **Generate vessel graphs and images**
```bash
python pathology/generate_synthetic_octa_images.py \
  --num_samples 100 \
  --out_dir data/vlm_dataset_demo \
  --workers -1 \
  --use_gan \
  --gan_config /path/to/gan/config.yml \
  --gan_epoch 150
```
Skip `--use_gan` for grayscale outputs. GAN configs/weights (and the `test.py` runner) live in the upstream OCTA-autosegmentation repo; to use GAN rendering, run this script inside that repo or copy it into that workspace.

2) **Build VLM pairs with pathology reasoning**
```bash
python pathology/generate_vlm_dataset.py \
  --num_samples 100 \
  --out_dir data/vlm_dataset_demo \
  --use_gan \
  --stage all
```

3) **(Optional) Post-process overlays/metadata**
```bash
python pathology/pathology_postprocess.py --dataset data/vlm_dataset_demo
```

Key outputs: `images/*.png`, `metadata.jsonl`, and `pairs.jsonl` (ShareGPT format). Adjust pathology ranges in `generate_vlm_dataset.py` as needed.

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
