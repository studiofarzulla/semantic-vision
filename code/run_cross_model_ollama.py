#!/usr/bin/env python3
"""
Cross-Model Replication via Ollama API
========================================
Runs the persona prime experiment across VLMs using Ollama's ROCm-accelerated backend.

Models:
- llava:13b (LLaVA-1.5-13B)
- llava:7b (LLaVA-1.5-7B, smaller alternative)

Author: Farzulla Research
"""

import argparse
import json
import logging
import os
import base64
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests
import numpy as np
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_IMAGE_DIR = str(Path.home() / "Resurrexi/data/coco/val2017")
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results"

# The 7 Context Primes from the Semantic-First Paper (EXACT MATCH with run_pilot.py)
PRIMES = {
    "P0_Neutral": "Analyze this image objectively. List the 3 most prominent objects, their geometric properties, and standard functions.",

    "P1_Chef": "You are a professional chef examining this scene for cooking-related possibilities. Identify the 3 most critical items for food preparation and list their affordances (what you can do with them).",

    "P2_Security": "You are a security professional assessing this space for vulnerabilities and tactical assets. Identify 3 objects that represent risks or defensive tools and their affordances.",

    "P3_Child": "Imagine you are a 4-year-old child. Identify 3 interesting things to play with in this scene and how you would use them.",

    "P4_Mobility": "You are navigating this space in a wheelchair. Identify 3 objects that either obstruct your path or enable your movement.",

    "P5_Urgent": "EMERGENCY: You have 30 seconds to find a tool for immediate survival. What do you see first and how do you use it?",

    "P6_Leisure": "You are casually exploring this space with absolutely no time pressure. What catches your eye for pure enjoyment or relaxation?"
}

JSON_FORMAT_INSTRUCTION = "\n\nProvide response in JSON format with keys: 'objects' (list of {id, name, affordance, reasoning})."

# Model configurations
OLLAMA_MODELS = {
    "llava": "llava:13b",
    "llava-7b": "llava:7b",
    "llama3.2-vision": "llama3.2-vision:11b",
}


# ==============================================================================
# LOGGING SETUP
# ==============================================================================

def setup_logging(output_dir: Path, model_name: str) -> logging.Logger:
    """Configure logging to file and console."""
    log_file = output_dir / f"{model_name}_ollama_run.log"

    logger = logging.getLogger(f"semantic_vision_{model_name}")
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ==============================================================================
# OLLAMA API
# ==============================================================================

def check_ollama_connection() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def check_model_available(model: str) -> bool:
    """Check if a specific model is available in Ollama."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any(m.get("name", "").startswith(model.split(":")[0]) for m in models)
    except:
        pass
    return False


def encode_image(image_path: Path) -> str:
    """Encode image to base64 for Ollama API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_ollama_inference(
    model: str,
    prompt: str,
    image_path: Path,
    temperature: float = 0.7,
    max_tokens: int = 512
) -> str:
    """Run inference on Ollama API with image."""

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [encode_image(image_path)],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120
    )

    if response.status_code == 200:
        return response.json().get("response", "")
    else:
        raise Exception(f"Ollama API error: {response.status_code} - {response.text}")


# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

def get_processed_images(output_file: Path) -> set:
    """Get set of already processed (image, prime) pairs for resume capability."""
    processed = set()
    if output_file.exists():
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed.add((entry.get('image_id'), entry.get('prime_id')))
                except:
                    continue
    return processed


def run_experiment(
    model_name: str,
    images_dir: Path,
    output_file: Path,
    limit: int = 500,
    resume: bool = True,
    logger: logging.Logger = None,
    image_list_file: Optional[Path] = None
):
    """Run the full experiment for a given model via Ollama."""

    ollama_model = OLLAMA_MODELS.get(model_name, model_name)

    logger.info(f"Starting experiment with {model_name} (Ollama: {ollama_model})")
    logger.info(f"Output file: {output_file}")

    # Check Ollama connection
    if not check_ollama_connection():
        logger.error("Ollama is not running! Start with: ollama serve")
        return

    # Check model availability
    if not check_model_available(ollama_model):
        logger.error(f"Model {ollama_model} not found. Pull with: ollama pull {ollama_model}")
        return

    logger.info(f"Using Ollama model: {ollama_model}")

    # Discover images - either from list file or directory scan
    if image_list_file and image_list_file.exists():
        logger.info(f"Loading images from list file: {image_list_file}")
        with open(image_list_file, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        # Convert to (base_dir, filename) tuples for consistent handling
        images = []
        image_dirs = {}
        for p in image_paths[:limit]:
            path = Path(p)
            if path.exists():
                images.append(path.name)
                image_dirs[path.name] = path.parent
            else:
                logger.warning(f"Image not found: {p}")
        logger.info(f"Loaded {len(images)} valid image paths from list")
    else:
        logger.info(f"Images directory: {images_dir}")
        images = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])[:limit]
        image_dirs = {img: images_dir for img in images}

    if not images:
        logger.error(f"No images found")
        return

    logger.info(f"Found {len(images)} images to process")

    # Resume logic
    processed = set()
    if resume:
        processed = get_processed_images(output_file)
        logger.info(f"Resuming: {len(processed)} entries already processed")

    # Processing loop
    total_pairs = len(images) * len(PRIMES)
    pbar = tqdm(total=total_pairs - len(processed), desc=f"{model_name} inference")

    with open(output_file, "a") as f:
        for img_name in images:
            img_path = image_dirs[img_name] / img_name

            for prime_id, prime_text in PRIMES.items():
                # Skip if already processed
                if (img_name, prime_id) in processed:
                    continue

                full_prompt = prime_text + JSON_FORMAT_INSTRUCTION

                start_time = time.time()
                try:
                    output = run_ollama_inference(
                        model=ollama_model,
                        prompt=full_prompt,
                        image_path=img_path,
                        temperature=0.7,
                        max_tokens=512
                    )

                    result = {
                        "image_id": img_name,
                        "prime_id": prime_id,
                        "prime_text": prime_text,
                        "raw_output": output,
                        "timestamp": time.time(),
                        "inference_time": time.time() - start_time,
                        "model_name": model_name,
                        "ollama_model": ollama_model
                    }

                    f.write(json.dumps(result) + "\n")
                    f.flush()

                    logger.debug(f"{img_name} / {prime_id}: {time.time() - start_time:.2f}s")

                except Exception as e:
                    logger.error(f"{img_name} / {prime_id}: FAILED - {e}")
                    error_result = {
                        "image_id": img_name,
                        "prime_id": prime_id,
                        "error": str(e),
                        "timestamp": time.time(),
                        "model_name": model_name
                    }
                    f.write(json.dumps(error_result) + "\n")
                    f.flush()

                pbar.update(1)

    pbar.close()
    logger.info(f"Experiment complete. Results saved to {output_file}")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-model replication via Ollama API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run LLaVA on original images
  python run_cross_model_ollama.py --model llava --image-list original_images.txt

  # Run on COCO images
  python run_cross_model_ollama.py --model llava --images-dir ~/data/coco/val2017 --limit 100
        """
    )

    parser.add_argument(
        "--model", type=str, default="llava",
        choices=list(OLLAMA_MODELS.keys()),
        help="Model to use"
    )
    parser.add_argument(
        "--images-dir", type=str, default=DEFAULT_IMAGE_DIR,
        help="Directory containing images"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--limit", type=int, default=500,
        help="Maximum number of images to process"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Don't resume from previous run"
    )
    parser.add_argument(
        "--image-list", type=str, default=None,
        help="File containing list of image paths (one per line)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup output directory
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    logger = setup_logging(output_dir, args.model)

    output_file = Path(args.output) if args.output else output_dir / f"{args.model}_ollama_results.jsonl"
    images_dir = Path(args.images_dir)
    image_list_file = Path(args.image_list) if args.image_list else None

    # Validate inputs
    if not image_list_file and not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return
    if image_list_file and not image_list_file.exists():
        logger.error(f"Image list file not found: {image_list_file}")
        return

    try:
        run_experiment(
            model_name=args.model,
            images_dir=images_dir,
            output_file=output_file,
            limit=args.limit,
            resume=not args.no_resume,
            logger=logger,
            image_list_file=image_list_file
        )
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
