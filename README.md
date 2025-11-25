# OCTA-SVR (Synthetic Vasculature Reasoning)

An anonymized release of the code used for the **Synthetic Vasculature Reasoning (SVR)** pipeline: pathology-aware OCTA synthesis plus LLaMA-Factory training/inference configs for VLM reasoning.

- Synthetic generation and text pairing scripts live in `pathology/` (forked from and compatible with [OCTA-autosegmentation](https://github.com/aiforvision/OCTA-autosegmentation)).
- LLaMA-Factory configs and inference helpers live in `llama_factory/`.
- Placeholder release of the **OCTA-100K-SVR** dataset: `https://zenodo.org/record/0000000` (replace with the real Zenodo record once available).

## Quickstart
```bash
git clone <this-repo>
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
You can skip `--use_gan` for grayscale outputs. The GAN config/checkpoints can be downloaded from the upstream OCTA-autosegmentation repo.

2) **Build VLM pairs with pathology reasoning**
```bash
python pathology/generate_vlm_dataset.py \
  --num_samples 100 \
  --out_dir data/vlm_dataset_demo \
  --use_gan \
  --stage all
```

3) **Post-process overlays/metadata (optional cleanup)**
```bash
python pathology/pathology_postprocess.py --dataset data/vlm_dataset_demo
```

The key outputs are `images/*.png`, `metadata.jsonl`, and `pairs.jsonl` (ShareGPT format). Adjust sampling ranges in `generate_vlm_dataset.py` for different pathology strengths.

## Training with LLaMA-Factory
Download/unzip `OCTA-100K-SVR` to `data/OCTA-100K-SVR/` so that `pairs_diversified.jsonl` and `images/` sit under that folder. Then run:
```bash
bash llama_factory/scripts/run_svr_training.sh
```
The script calls `llamafactory-cli` with `llama_factory/configs/svr_qwen3vl_full.yaml` and `llama_factory/data/dataset_info.json`. Edit the config to point to a different base model or change batch size/epochs.

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
- `pathology/`: Synthetic vessel/pathology simulation, CoT pairing, and cleanup utilities.
- `pathology/vessel_graph_generation/`: Graph-growth simulator and rasterization helpers.
- `llama_factory/`: Training config, dataset descriptors, inference helpers, and runnable scripts.
- `requirements.txt`: Minimal dependencies (GAN weights/models are not included).

## Notes
- Paths in this release are relative; no cluster-specific directories remain.
- Pretrained GAN weights and any clinical data are intentionally excluded. Use the upstream OCTA-autosegmentation repo or your own models for GAN inference.
- Replace the placeholder Zenodo link with the final public record when available.
