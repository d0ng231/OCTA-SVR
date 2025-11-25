#!/usr/bin/env python
"""
OCTA Image Inference Script
Supports both single image and batch inference using trained VLM model

Usage:
    # Single image inference
    python inference_octa_CoT.py --model_path /path/to/base_or_merged_model --image /path/to/image.png
    
    # Batch inference on a folder
    python inference_octa_CoT.py --model_path /path/to/base_or_merged_model --image_dir /path/to/images/ --output results.jsonl
    
    # With LoRA adapter (recommended for your fine-tuned thinking model)
    python inference_octa_CoT.py \
        --model_path Qwen/Qwen3-VL-30B-A3B-Thinking \
        --adapter /path/to/lora/checkpoint-XXXX \
    --template qwen3_vl \
        --image /path/to/image.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import re

# Add parent directory to path to import llamafactory
sys.path.insert(0, str(Path(__file__).parent.parent))

from llamafactory.chat import ChatModel
def _has_weights(dir_path: Path) -> bool:
    """Return True if the directory contains recognizable HF weight files."""
    if not dir_path.exists() or not dir_path.is_dir():
        return False
    # Common single-file names
    if (dir_path / "pytorch_model.bin").exists():
        return True
    if (dir_path / "model.safetensors").exists():
        return True
    # Sharded weights are referenced by an index file
    if (dir_path / "model.safetensors.index.json").exists():
        return True
    # Fallback: look for shard pattern
    shards = list(dir_path.glob("model-*-of-*.safetensors"))
    return len(shards) > 0


def resolve_model_path(model_path: str) -> str:
    """
    Resolve an effective model path that actually contains weight files.

    Behavior:
    - If model_path already contains weights, return as-is.
    - Else, search subdirectories matching 'checkpoint-*' and pick the highest step
      that contains weights. If found, return that subdir. Otherwise return original.
    """
    root = Path(model_path)
    if _has_weights(root):
        return str(root)

    # Search for checkpoint-* subdirectories and select the latest by step number
    candidates: list[tuple[int, Path]] = []
    for sub in root.glob("checkpoint-*"):
        if not sub.is_dir():
            continue
        m = re.search(r"checkpoint-(\d+)$", sub.name)
        if not m:
            continue
        step = int(m.group(1))
        if _has_weights(sub):
            candidates.append((step, sub))

    if candidates:
        # Pick the largest step
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0][1]
        print(f"[auto] Resolved model_path to latest checkpoint with weights: {best}")
        return str(best)

    # Nothing found; return original
    return str(root)



def setup_model(
    model_path: str,
    adapter_path: str | None = None,
    template: str = "qwen3_vl",
    enable_thinking: bool = False,
    infer_backend: str = "huggingface",
    temperature: float = 0.1,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
) -> ChatModel:
    """
    Initialize the ChatModel with the trained checkpoint
    
    Args:
        model_path: Path to the trained model checkpoint
    
    Returns:
        ChatModel instance
    """
    infer_args = {
        "model_name_or_path": model_path,
        "adapter_name_or_path": adapter_path,
        "template": template,
        "enable_thinking": enable_thinking,
        "trust_remote_code": True,
        "image_max_pixels": 262144,
        "video_max_pixels": 16384,
        "infer_backend": infer_backend,
        # decoding
        "do_sample": temperature > 0,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    
    print("=" * 80)
    print("Initializing model...")
    print(f"Model path: {model_path}")
    print(f"Adapter   : {adapter_path if adapter_path else 'None'}")
    print(f"Template  : {template} | Thinking: {enable_thinking}")
    print(f"Backend   : {infer_backend}")
    print("=" * 80)
    
    return ChatModel(infer_args)


def inference_single_image(
    chat_model: ChatModel,
    image_path: str,
    question: str = "What do you see in this OCTA image?",
    stream: bool = True
) -> str:
    """
    Run inference on a single image
    
    Args:
        chat_model: ChatModel instance
        image_path: Path to the image file
        question: Question to ask about the image
        stream: Whether to stream the output
    
    Returns:
        Model response text
    """
    messages = [{"role": "user", "content": question}]
    images = [image_path]
    
    print(f"\nImage: {Path(image_path).name}")
    print(f"Question: {question}")
    print("-" * 80)
    
    if stream:
        print("Response: ", end="", flush=True)
        full_response = ""
        for token in chat_model.stream_chat(messages=messages, images=images):
            print(token, end="", flush=True)
            full_response += token
        print("\n")
        return full_response
    else:
        responses = chat_model.chat(messages=messages, images=images)
        response_text = responses[0].response_text if responses else ""
        print(f"Response: {response_text}\n")
        return response_text


def inference_batch(
    chat_model: ChatModel,
    image_dir: str,
    output_file: str,
    question: str = "What do you see in this OCTA image?",
    image_extensions: List[str] = None
) -> None:
    """
    Run batch inference on all images in a directory
    
    Args:
        chat_model: ChatModel instance
        image_dir: Directory containing images
        output_file: Path to save results (JSONL format)
        question: Question to ask about each image
        image_extensions: List of valid image extensions
    """
    if image_extensions is None:
        image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]
    
    # Find all image files
    image_dir_path = Path(image_dir)
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir_path.glob(f"*{ext}")))
        image_files.extend(list(image_dir_path.glob(f"*{ext.upper()}")))
    
    image_files = sorted(set(image_files))
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    print(f"\nFound {len(image_files)} image files")
    print(f"Output will be saved to: {output_file}")
    print("=" * 80)
    
    # Run batch inference
    results = []
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for img_path in tqdm(image_files, desc="Processing images"):
            messages = [{"role": "user", "content": question}]
            images = [str(img_path)]
            
            try:
                # Get response (non-streaming for batch)
                responses = chat_model.chat(messages=messages, images=images)
                response_text = responses[0].response_text if responses else ""
                
                result = {
                    "image_path": str(img_path),
                    "image_name": img_path.name,
                    "question": question,
                    "response": response_text
                }
                
                # Write result immediately
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
                
                results.append(result)
                
                # Print preview
                preview = response_text[:150] + "..." if len(response_text) > 150 else response_text
                tqdm.write(f"✓ {img_path.name}: {preview}")
                
            except Exception as e:
                error_msg = f"Error processing {img_path}: {str(e)}"
                tqdm.write(f"✗ {error_msg}")
                
                # Save error to output
                error_result = {
                    "image_path": str(img_path),
                    "image_name": img_path.name,
                    "question": question,
                    "error": str(e)
                }
                f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                f.flush()
                continue
    
    print("\n" + "=" * 80)
    print(f"Batch inference completed!")
    print(f"Successfully processed: {len(results)}/{len(image_files)} images")
    print(f"Results saved to: {output_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="OCTA Image Inference - Single or Batch mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image inference
  python inference_octa_CoT.py --model_path /path/to/qwen3vl_svr_checkpoint --image /path/to/image.png
  
    # Batch inference
    python inference_octa_CoT.py --model_path /path/to/qwen3vl_svr_checkpoint --image_dir /path/to/images/ --output results.jsonl
  
  # Qwen3 thinking base with LoRA adapter
  python inference_octa_CoT.py --model_path Qwen/Qwen3-VL-30B-A3B-Thinking \
      --adapter /path/to/lora/checkpoint-500 \
      --template qwen3_vl --image /path/to/image.png
  
    # Note: If you pass a training root (e.g., .../sft/) without weights at top-level,
    #       this script will automatically resolve to the latest checkpoint-* that
    #       contains weights.

    # Custom question
  python inference_octa_CoT.py --model_path /path/to/model --image /path/to/image.png --question "Describe the vessels in this image"
        """
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (required)"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Optional LoRA adapter path (LLaMA-Factory output_dir or a checkpoint-XXXX directory)."
    )
    parser.add_argument(
        "--template",
        type=str,
        default="qwen3_vl",
        help="Chat template: qwen3_vl (thinking) or qwen2_vl, etc."
    )
    parser.add_argument(
        "--disable_thinking",
        action="store_false",
        help="Disable thinking mode for reasoning models (sets enable_thinking=False)."
    )
    parser.add_argument(
        "--infer_backend",
        type=str,
        default="huggingface",
        choices=["huggingface", "vllm", "sglang"],
        help="Inference backend engine."
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=str,
        help="Path to a single image file"
    )
    input_group.add_argument(
        "--image_dir",
        type=str,
        help="Path to directory containing images for batch inference"
    )
    
    # Inference arguments
    parser.add_argument(
        "--question",
        type=str,
        default="What do you see in this OCTA image?",
        help="Question to ask about the image(s)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference_results.jsonl",
        help="Output file for batch inference results (JSONL format)"
    )
    parser.add_argument(
        "--no_stream",
        action="store_true",
        help="Disable streaming output for single image inference"
    )
    parser.add_argument(
        "--image_extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"],
        help="Valid image file extensions for batch inference"
    )
    
    args = parser.parse_args()
    
    # Resolve model path to ensure we point to a directory containing weights
    effective_model_path = resolve_model_path(args.model_path)

    # Initialize model (with optional LoRA adapter and thinking support)
    chat_model = setup_model(
        model_path=effective_model_path,
        adapter_path=args.adapter,
        template=args.template,
        enable_thinking=(not args.disable_thinking),
        infer_backend=args.infer_backend,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    
    # Run inference
    if args.image:
        # Single image mode
        if not Path(args.image).exists():
            print(f"Error: Image file not found: {args.image}")
            sys.exit(1)
        
        response = inference_single_image(
            chat_model=chat_model,
            image_path=args.image,
            question=args.question,
            stream=not args.no_stream
        )
        
    else:
        # Batch mode
        if not Path(args.image_dir).exists():
            print(f"Error: Directory not found: {args.image_dir}")
            sys.exit(1)
        
        inference_batch(
            chat_model=chat_model,
            image_dir=args.image_dir,
            output_file=args.output,
            question=args.question,
            image_extensions=args.image_extensions
        )


if __name__ == "__main__":
    main()
