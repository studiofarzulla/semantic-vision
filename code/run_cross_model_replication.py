#!/usr/bin/env python3
"""
Cross-Model Replication: Semantic-First Vision Experiment

Runs the persona prime experiment across multiple VLMs for replication:
- LLaVA-1.5-13B (llava-hf/llava-1.5-13b-hf)
- BLIP-2 (Salesforce/blip2-opt-2.7b)
- CogVLM (THUDM/cogvlm-chat-hf)

Hardware: PurrPower with 7900 XTX (24GB) + 7800 XT (16GB), 128GB RAM
Uses transformers with bfloat16 for AMD ROCm compatibility.

Author: Farzulla Research
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import itertools

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================

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
MODEL_CONFIGS = {
    "llava": {
        "model_id": "llava-hf/llava-1.5-13b-hf",
        "processor_class": "LlavaProcessor",
        "model_class": "LlavaForConditionalGeneration",
        "max_memory": {"0": "24GB"},  # Fits on 7900 XTX
        "requires_image_token": True,
    },
    "blip2": {
        "model_id": "Salesforce/blip2-opt-2.7b",
        "processor_class": "Blip2Processor",
        "model_class": "Blip2ForConditionalGeneration",
        "max_memory": {"0": "16GB"},  # Smaller, fits easily
        "requires_image_token": False,
    },
    "cogvlm": {
        "model_id": "THUDM/cogvlm-chat-hf",
        "processor_class": "AutoProcessor",
        "model_class": "AutoModelForCausalLM",
        "max_memory": {"0": "24GB"},
        "requires_image_token": False,
        "trust_remote_code": True,
    },
}

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

def setup_logging(output_dir: Path, model_name: str) -> logging.Logger:
    """Configure logging to file and console."""
    log_file = output_dir / f"{model_name}_run.log"

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
# MODEL LOADING
# ==============================================================================

def load_model(model_name: str, logger: logging.Logger) -> Tuple[Any, Any]:
    """
    Load VLM model and processor based on model name.
    Returns (model, processor) tuple.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]
    logger.info(f"Loading model: {config['model_id']}")

    # Import transformers components
    from transformers import (
        LlavaProcessor, LlavaForConditionalGeneration,
        Blip2Processor, Blip2ForConditionalGeneration,
        AutoProcessor, AutoModelForCausalLM,
        BitsAndBytesConfig
    )

    # Common loading kwargs
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }

    if model_name == "llava":
        processor = LlavaProcessor.from_pretrained(config['model_id'])
        model = LlavaForConditionalGeneration.from_pretrained(
            config['model_id'],
            **load_kwargs
        )

    elif model_name == "blip2":
        processor = Blip2Processor.from_pretrained(config['model_id'])
        model = Blip2ForConditionalGeneration.from_pretrained(
            config['model_id'],
            **load_kwargs
        )

    elif model_name == "cogvlm":
        processor = AutoProcessor.from_pretrained(
            config['model_id'],
            trust_remote_code=config.get('trust_remote_code', False)
        )
        model = AutoModelForCausalLM.from_pretrained(
            config['model_id'],
            trust_remote_code=config.get('trust_remote_code', False),
            **load_kwargs
        )

    model.eval()
    logger.info(f"Model loaded successfully. Device: {next(model.parameters()).device}")

    return model, processor


# ==============================================================================
# INFERENCE FUNCTIONS
# ==============================================================================

def format_prompt(model_name: str, prime_text: str, processor: Any) -> str:
    """Format prompt according to model-specific requirements."""
    full_prompt = prime_text + JSON_FORMAT_INSTRUCTION

    if model_name == "llava":
        # LLaVA uses <image> token
        return f"<image>\nUSER: {full_prompt}\nASSISTANT:"

    elif model_name == "blip2":
        # BLIP-2 simple question format
        return f"Question: {full_prompt} Answer:"

    elif model_name == "cogvlm":
        # CogVLM chat format
        return f"Question: {full_prompt}\nAnswer:"

    return full_prompt


def run_inference(
    model: Any,
    processor: Any,
    model_name: str,
    image: Image.Image,
    prompt: str,
    temperature: float = 0.7,
    max_new_tokens: int = 512
) -> str:
    """Run inference on a single image with prompt."""

    # Prepare inputs based on model type
    if model_name == "llava":
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(model.device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        # Decode, skipping input tokens
        output = processor.decode(
            output_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

    elif model_name == "blip2":
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )

        output = processor.decode(output_ids[0], skip_special_tokens=True)
        # BLIP-2 sometimes repeats the prompt
        if prompt in output:
            output = output.split(prompt)[-1].strip()

    elif model_name == "cogvlm":
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )

        output = processor.decode(
            output_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

    return output.strip()


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
    """Run the full experiment for a given model."""

    logger.info(f"Starting experiment with {model_name}")
    logger.info(f"Output file: {output_file}")

    # Load model
    model, processor = load_model(model_name, logger)

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

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.error(f"Failed to load {img_name}: {e}")
                continue

            for prime_id, prime_text in PRIMES.items():
                # Skip if already processed
                if (img_name, prime_id) in processed:
                    continue

                # Format prompt
                prompt = format_prompt(model_name, prime_text, processor)

                start_time = time.time()
                try:
                    output = run_inference(
                        model, processor, model_name,
                        image, prompt,
                        temperature=0.7,
                        max_new_tokens=512
                    )

                    result = {
                        "image_id": img_name,
                        "prime_id": prime_id,
                        "prime_text": prime_text,
                        "raw_output": output,
                        "timestamp": time.time(),
                        "inference_time": time.time() - start_time,
                        "model_name": model_name,
                        "model_id": MODEL_CONFIGS[model_name]['model_id']
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

                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    pbar.close()
    logger.info(f"Experiment complete. Results saved to {output_file}")


# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def jaccard_words(str1: str, str2: str) -> float:
    """Word-level Jaccard similarity."""
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    if not a or not b:
        return 0.0
    return len(a.intersection(b)) / len(a.union(b))


def jaccard_objects(set1: set, set2: set) -> float:
    """Object-level Jaccard similarity."""
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))


def extract_objects(raw_output: str) -> set:
    """Extract object names from raw JSON output."""
    try:
        # Clean markdown formatting
        clean = raw_output.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        return set([
            obj.get('name', '').lower().strip()
            for obj in parsed.get('objects', [])
            if obj.get('name', '').strip()
        ])
    except:
        return set()


def extract_text(raw_output: str) -> str:
    """Extract full text from raw JSON output."""
    try:
        clean = raw_output.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        return " ".join([
            f"{obj.get('name', '')} {obj.get('affordance', '')} {obj.get('reasoning', '')}"
            for obj in parsed.get('objects', [])
        ]).lower()
    except:
        return raw_output.lower()


def analyze_results(
    results_files: List[Path],
    baseline_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Analyze results from multiple models.
    Computes Jaccard similarity, cross-model comparison, and statistical tests.
    """
    from scipy import stats

    analysis = {
        "models": {},
        "cross_model_comparison": {},
        "statistical_tests": {}
    }

    # Load all results
    all_data = {}
    for rf in results_files:
        model_name = rf.stem.replace("_results", "")
        data = []
        with open(rf, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "error" not in entry:
                        entry['objects'] = extract_objects(entry['raw_output'])
                        entry['text'] = extract_text(entry['raw_output'])
                        data.append(entry)
                except:
                    continue
        all_data[model_name] = data
        logger.info(f"Loaded {len(data)} entries from {model_name}")

    # Per-model Jaccard analysis
    for model_name, data in all_data.items():
        # Group by image
        by_image = {}
        for entry in data:
            img = entry['image_id']
            if img not in by_image:
                by_image[img] = {}
            by_image[img][entry['prime_id']] = entry

        word_sims = []
        obj_sims = []

        for img, primes in by_image.items():
            prime_ids = list(primes.keys())
            for p1, p2 in itertools.combinations(prime_ids, 2):
                word_sims.append(jaccard_words(primes[p1]['text'], primes[p2]['text']))
                obj_sims.append(jaccard_objects(primes[p1]['objects'], primes[p2]['objects']))

        analysis["models"][model_name] = {
            "n_entries": len(data),
            "n_images": len(by_image),
            "jaccard_word": {
                "mean": float(np.mean(word_sims)),
                "std": float(np.std(word_sims)),
                "median": float(np.median(word_sims)),
                "n_pairs": len(word_sims)
            },
            "jaccard_object": {
                "mean": float(np.mean(obj_sims)),
                "std": float(np.std(obj_sims)),
                "median": float(np.median(obj_sims)),
                "n_pairs": len(obj_sims)
            }
        }

        # Permutation test: mean < 0.5
        t_word, p_word = stats.ttest_1samp(word_sims, 0.5)
        t_obj, p_obj = stats.ttest_1samp(obj_sims, 0.5)

        analysis["models"][model_name]["significance"] = {
            "word_ttest": {
                "t_statistic": float(t_word),
                "p_value_two_sided": float(p_word),
                "p_value_one_sided": float(p_word / 2) if t_word < 0 else 1.0,
                "significant_001": p_word / 2 < 0.01 if t_word < 0 else False
            },
            "object_ttest": {
                "t_statistic": float(t_obj),
                "p_value_two_sided": float(p_obj),
                "p_value_one_sided": float(p_obj / 2) if t_obj < 0 else 1.0,
                "significant_001": p_obj / 2 < 0.01 if t_obj < 0 else False
            }
        }

        logger.info(f"{model_name}: word_jaccard={np.mean(word_sims):.4f}, obj_jaccard={np.mean(obj_sims):.4f}")

    # Load baseline (Qwen-VL) if provided
    if baseline_file and baseline_file.exists():
        baseline_data = []
        with open(baseline_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "error" not in entry:
                        entry['objects'] = extract_objects(entry['raw_output'])
                        entry['text'] = extract_text(entry['raw_output'])
                        baseline_data.append(entry)
                except:
                    continue

        all_data['qwen_baseline'] = baseline_data
        logger.info(f"Loaded {len(baseline_data)} baseline entries")

        # Cross-model comparison with baseline
        baseline_by_img = {}
        for entry in baseline_data:
            img = entry['image_id']
            if img not in baseline_by_img:
                baseline_by_img[img] = {}
            baseline_by_img[img][entry['prime_id']] = entry

        for model_name, data in all_data.items():
            if model_name == 'qwen_baseline':
                continue

            model_by_img = {}
            for entry in data:
                img = entry['image_id']
                if img not in model_by_img:
                    model_by_img[img] = {}
                model_by_img[img][entry['prime_id']] = entry

            # Compare same (image, prime) pairs
            cross_word = []
            cross_obj = []

            common_imgs = set(baseline_by_img.keys()) & set(model_by_img.keys())
            for img in common_imgs:
                common_primes = set(baseline_by_img[img].keys()) & set(model_by_img[img].keys())
                for prime in common_primes:
                    base_entry = baseline_by_img[img][prime]
                    model_entry = model_by_img[img][prime]

                    cross_word.append(jaccard_words(base_entry['text'], model_entry['text']))
                    cross_obj.append(jaccard_objects(base_entry['objects'], model_entry['objects']))

            analysis["cross_model_comparison"][f"{model_name}_vs_qwen"] = {
                "n_common_pairs": len(cross_word),
                "jaccard_word": {
                    "mean": float(np.mean(cross_word)) if cross_word else 0,
                    "std": float(np.std(cross_word)) if cross_word else 0
                },
                "jaccard_object": {
                    "mean": float(np.mean(cross_obj)) if cross_obj else 0,
                    "std": float(np.std(cross_obj)) if cross_obj else 0
                }
            }

            logger.info(f"{model_name} vs qwen: {len(cross_word)} pairs, word={np.mean(cross_word) if cross_word else 0:.4f}")

    # Cross-model consistency test
    if len(all_data) >= 2:
        model_means = []
        for model_name, stats in analysis["models"].items():
            model_means.append({
                'model': model_name,
                'word_mean': stats['jaccard_word']['mean'],
                'obj_mean': stats['jaccard_object']['mean']
            })

        analysis["cross_model_consistency"] = {
            "models_compared": [m['model'] for m in model_means],
            "word_jaccard_variance": float(np.var([m['word_mean'] for m in model_means])),
            "object_jaccard_variance": float(np.var([m['obj_mean'] for m in model_means]))
        }

    # Save analysis
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Analysis saved to {output_file}")

    return analysis


def print_analysis_summary(analysis: Dict[str, Any]):
    """Print a formatted summary of the analysis."""
    print("\n" + "=" * 70)
    print("CROSS-MODEL REPLICATION ANALYSIS SUMMARY")
    print("=" * 70)

    for model, stats in analysis.get("models", {}).items():
        print(f"\n[{model.upper()}]")
        print(f"  Entries: {stats['n_entries']} | Images: {stats['n_images']}")
        print(f"  Word-level Jaccard:   mean={stats['jaccard_word']['mean']:.4f} (SD={stats['jaccard_word']['std']:.4f})")
        print(f"  Object-level Jaccard: mean={stats['jaccard_object']['mean']:.4f} (SD={stats['jaccard_object']['std']:.4f})")

        if 'significance' in stats:
            sig_w = stats['significance']['word_ttest']
            sig_o = stats['significance']['object_ttest']
            star_w = "***" if sig_w['significant_001'] else ""
            star_o = "***" if sig_o['significant_001'] else ""
            print(f"  H-S1 (word):   t={sig_w['t_statistic']:.2f}, p={sig_w['p_value_one_sided']:.4f} {star_w}")
            print(f"  H-S1 (object): t={sig_o['t_statistic']:.2f}, p={sig_o['p_value_one_sided']:.4f} {star_o}")

    if analysis.get("cross_model_comparison"):
        print("\n[CROSS-MODEL COMPARISON]")
        for comp, stats in analysis["cross_model_comparison"].items():
            print(f"  {comp}:")
            print(f"    Common pairs: {stats['n_common_pairs']}")
            print(f"    Word Jaccard: {stats['jaccard_word']['mean']:.4f} (SD={stats['jaccard_word']['std']:.4f})")
            print(f"    Object Jaccard: {stats['jaccard_object']['mean']:.4f} (SD={stats['jaccard_object']['std']:.4f})")

    if analysis.get("cross_model_consistency"):
        cons = analysis["cross_model_consistency"]
        print("\n[CROSS-MODEL CONSISTENCY]")
        print(f"  Word Jaccard variance across models: {cons['word_jaccard_variance']:.6f}")
        print(f"  Object Jaccard variance across models: {cons['object_jaccard_variance']:.6f}")

    print("\n" + "=" * 70)


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-model replication of Semantic-First Vision experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run LLaVA on COCO images
  python run_cross_model_replication.py --model llava --images-dir ~/data/coco/val2017

  # Run BLIP-2 with custom output
  python run_cross_model_replication.py --model blip2 --output results/blip2_results.jsonl

  # Analyze all results and compare with Qwen baseline
  python run_cross_model_replication.py --analyze --baseline qwen_results.jsonl

  # Run all models sequentially
  python run_cross_model_replication.py --model all --limit 100
        """
    )

    parser.add_argument(
        "--model", type=str, default="llava",
        choices=["llava", "blip2", "cogvlm", "all"],
        help="Model to use (or 'all' to run sequentially)"
    )
    parser.add_argument(
        "--images-dir", type=str, default=DEFAULT_IMAGE_DIR,
        help="Directory containing images"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL file path (default: results/<model>_results.jsonl)"
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
        help="File containing list of image paths (one per line), overrides --images-dir"
    )

    # Analysis options
    parser.add_argument(
        "--analyze", action="store_true",
        help="Run analysis on existing results instead of inference"
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Baseline results file (Qwen-VL) for cross-model comparison"
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory containing result files for analysis"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup output directory
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    if args.analyze:
        # Analysis mode
        logger = setup_logging(output_dir, "analysis")

        # Find result files
        results_dir = Path(args.results_dir) if args.results_dir else output_dir
        result_files = list(results_dir.glob("*_results.jsonl"))

        if not result_files:
            logger.error(f"No result files found in {results_dir}")
            return

        logger.info(f"Found {len(result_files)} result files")

        baseline = Path(args.baseline) if args.baseline else None
        if not baseline and (output_dir.parent / "semantic_vision_pilot_results_qwen3_30b.jsonl").exists():
            baseline = output_dir.parent / "semantic_vision_pilot_results_qwen3_30b.jsonl"
            logger.info(f"Using default baseline: {baseline}")

        analysis = analyze_results(
            result_files,
            baseline_file=baseline,
            output_file=output_dir / "cross_model_analysis.json",
            logger=logger
        )

        print_analysis_summary(analysis)

    else:
        # Inference mode
        models_to_run = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]

        for model_name in models_to_run:
            logger = setup_logging(output_dir, model_name)

            output_file = Path(args.output) if args.output else output_dir / f"{model_name}_results.jsonl"
            images_dir = Path(args.images_dir)
            image_list_file = Path(args.image_list) if args.image_list else None

            # Validate inputs
            if not image_list_file and not images_dir.exists():
                logger.error(f"Images directory not found: {images_dir}")
                continue
            if image_list_file and not image_list_file.exists():
                logger.error(f"Image list file not found: {image_list_file}")
                continue

            try:
                run_experiment(
                    model_name=model_name,
                    images_dir=images_dir,
                    output_file=output_file,
                    limit=args.limit,
                    resume=not args.no_resume,
                    logger=logger,
                    image_list_file=image_list_file
                )
            except Exception as e:
                logger.error(f"Experiment failed for {model_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

            # Cleanup between models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
