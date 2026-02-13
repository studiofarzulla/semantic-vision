#!/usr/bin/env python3
"""
Stochastic Baseline Experiment - Semantic Vision Paper

Critical experiment to address Wolf's main critique:
"How much of the observed drift is context-dependent vs. stochastic variation?"

Design:
- 50 images (stratified sample from original 479)
- 7 context primes
- 5 random seeds per (image, prime) pair
- 4 temperature levels: [0.0, 0.3, 0.7, 1.0]

Key metrics:
- Within-prime variance: σ²_within (same prime, different seeds)
- Cross-prime variance: σ²_cross (different primes, same seed)
- Variance ratio: σ²_cross / σ²_within (should be >>1 to claim context effect)
- Effect size: η² for prime as factor

Author: Farzulla Research
Date: January 2026
"""

import base64
import json
import os
import time
import argparse
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

# API client
from openai import OpenAI

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
CODE_DIR = Path(__file__).parent
DEFAULT_IMAGE_DIR = Path.home() / "Resurrexi/data/coco/val2017"
ORIGINAL_RESULTS = PROJECT_DIR / "semantic_vision_pilot_results_qwen3_30b.jsonl"
OUTPUT_DIR = CODE_DIR / "results" / "stochastic_baseline"

# Prometheus endpoint (local dual GPU setup)
API_BASE = "http://localhost:1234/v1"
API_KEY = "lm-studio"
MODEL_NAME = "qwen/qwen3-vl-30b:2"

# Experiment parameters
N_IMAGES = 50  # Stratified sample
N_SEEDS = 5
TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
MAX_TOKENS = 512

# Context primes (same as original)
PRIMES = {
    "P0_Neutral": "Analyze this image objectively. List the 3 most prominent objects, their geometric properties, and standard functions.",
    "P1_Chef": "You are a professional chef examining this scene for cooking-related possibilities. Identify the 3 most critical items for food preparation and list their affordances (what you can do with them).",
    "P2_Security": "You are a security professional assessing this space for vulnerabilities and tactical assets. Identify 3 objects that represent risks or defensive tools and their affordances.",
    "P3_Child": "Imagine you are a 4-year-old child. Identify 3 interesting things to play with in this scene and how you would use them.",
    "P4_Mobility": "You are navigating this space in a wheelchair. Identify 3 objects that either obstruct your path or enable your movement.",
    "P5_Urgent": "EMERGENCY: You have 30 seconds to find a tool for immediate survival. What do you see first and how do you use it?",
    "P6_Leisure": "You are casually exploring this space with absolutely no time pressure. What catches your eye for pure enjoyment or relaxation?"
}


def encode_image(image_path):
    """Encodes image to base64 for API transmission."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_original_images():
    """Get list of images used in original experiment for stratified sampling."""
    images = set()
    if ORIGINAL_RESULTS.exists():
        with open(ORIGINAL_RESULTS, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if 'image_id' in entry and 'error' not in entry:
                        images.add(entry['image_id'])
                except:
                    continue
    return sorted(list(images))


def stratified_sample(images, n_sample, seed=42):
    """
    Stratified sample of images to ensure diversity.
    Groups by first digit of image ID (rough scene category proxy).
    """
    random.seed(seed)

    # Group by first digit of numeric part
    groups = defaultdict(list)
    for img in images:
        # Extract numeric part
        num_part = ''.join(filter(str.isdigit, img))
        if num_part:
            first_digit = num_part[0]
            groups[first_digit].append(img)

    # Sample proportionally from each group
    sampled = []
    total = len(images)
    for group_id, group_images in sorted(groups.items()):
        group_size = len(group_images)
        n_from_group = max(1, int(n_sample * group_size / total))
        sampled.extend(random.sample(group_images, min(n_from_group, len(group_images))))

    # If we need more, sample randomly from remaining
    if len(sampled) < n_sample:
        remaining = [img for img in images if img not in sampled]
        needed = n_sample - len(sampled)
        sampled.extend(random.sample(remaining, min(needed, len(remaining))))

    # If we have too many, trim
    if len(sampled) > n_sample:
        sampled = random.sample(sampled, n_sample)

    return sorted(sampled)


def run_inference(client, image_path, prime_text, temperature, seed):
    """Run single inference with specific temperature and seed."""
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prime_text}\n\nProvide response in JSON format with keys: 'objects' (list of {{id, name, affordance, reasoning}})."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        max_tokens=MAX_TOKENS,
        temperature=temperature,
        seed=seed  # Fixed seed for reproducibility
    )

    return response.choices[0].message.content


def load_checkpoint(output_file):
    """Load checkpoint to resume interrupted experiment."""
    completed = set()
    if output_file.exists():
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    key = (entry['image_id'], entry['prime_id'], entry['temperature'], entry['seed'])
                    completed.add(key)
                except:
                    continue
    return completed


def main():
    parser = argparse.ArgumentParser(description="Stochastic Baseline Experiment")
    parser.add_argument("--images", type=str, default=str(DEFAULT_IMAGE_DIR),
                       help="Path to image directory")
    parser.add_argument("--n-images", type=int, default=N_IMAGES,
                       help="Number of images to sample")
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS,
                       help="Number of seeds per (image, prime, temp) combination")
    parser.add_argument("--temperatures", type=float, nargs='+', default=TEMPERATURES,
                       help="Temperature values to test")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    parser.add_argument("--api-base", type=str, default=API_BASE,
                       help="API base URL")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print experiment plan without running")
    args = parser.parse_args()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"stochastic_baseline_m2_{timestamp}.jsonl"

    # Get original images
    original_images = get_original_images()
    if not original_images:
        print("[-] Could not find original experiment images.")
        print(f"    Expected: {ORIGINAL_RESULTS}")
        return

    print(f"[*] Found {len(original_images)} images from original experiment")

    # Stratified sample - MODEL 2 TAKES BACK HALF ONLY
    all_sampled = stratified_sample(original_images, args.n_images)
    sampled_images = all_sampled[25:]  # Back half (images 26-50)
    print(f"[*] Model 2: Processing back half ({len(sampled_images)} images)")

    # Calculate total runs
    n_primes = len(PRIMES)
    n_temps = len(args.temperatures)
    n_seeds = args.n_seeds
    total_runs = len(sampled_images) * n_primes * n_temps * n_seeds

    print(f"\n{'='*60}")
    print("STOCHASTIC BASELINE EXPERIMENT")
    print(f"{'='*60}")
    print(f"  Images:        {len(sampled_images)}")
    print(f"  Primes:        {n_primes}")
    print(f"  Temperatures:  {args.temperatures}")
    print(f"  Seeds/combo:   {n_seeds}")
    print(f"  Total runs:    {total_runs}")
    print(f"  Estimated time: ~{total_runs * 3 / 3600:.1f} hours @ 3s/run")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("[!] Dry run - not executing inference")
        print(f"    Sample images: {sampled_images[:5]}...")
        return

    # Connect to API
    print(f"[*] Connecting to {args.api_base}...")
    try:
        client = OpenAI(base_url=args.api_base, api_key=API_KEY)
        client.models.list()
        print("[+] Connection successful")
    except Exception as e:
        print(f"[-] Connection failed: {e}")
        return

    # Check for resume
    completed = set()
    if args.resume:
        # Find most recent output file
        existing = sorted(OUTPUT_DIR.glob("stochastic_baseline_m2_*.jsonl"))
        if existing:
            output_file = existing[-1]
            completed = load_checkpoint(output_file)
            print(f"[*] Resuming from {output_file.name}")
            print(f"    {len(completed)} runs already completed")

    # Build run queue
    runs = []
    for img in sampled_images:
        for prime_id, prime_text in PRIMES.items():
            for temp in args.temperatures:
                for seed in range(n_seeds):
                    key = (img, prime_id, temp, seed)
                    if key not in completed:
                        runs.append({
                            'image_id': img,
                            'prime_id': prime_id,
                            'prime_text': prime_text,
                            'temperature': temp,
                            'seed': seed
                        })

    print(f"[*] {len(runs)} runs remaining")

    # Image directory
    img_dir = Path(args.images)
    if not img_dir.exists():
        print(f"[-] Image directory not found: {img_dir}")
        return

    # Run experiment
    errors = 0
    with open(output_file, 'a') as f:
        for i, run in enumerate(runs):
            img_path = img_dir / run['image_id']
            if not img_path.exists():
                print(f"    [-] Image not found: {run['image_id']}")
                errors += 1
                continue

            start_time = time.time()
            try:
                output = run_inference(
                    client,
                    img_path,
                    run['prime_text'],
                    run['temperature'],
                    run['seed']
                )

                result = {
                    'image_id': run['image_id'],
                    'prime_id': run['prime_id'],
                    'temperature': run['temperature'],
                    'seed': run['seed'],
                    'raw_output': output,
                    'inference_time': time.time() - start_time,
                    'timestamp': time.time()
                }

                f.write(json.dumps(result) + "\n")
                f.flush()

                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    remaining = len(runs) - i - 1
                    eta = remaining * (elapsed / (i + 1)) / 3600
                    print(f"    [{i+1}/{len(runs)}] {run['prime_id']} T={run['temperature']} S={run['seed']} "
                          f"({result['inference_time']:.1f}s) ETA: {eta:.1f}h")

            except Exception as e:
                print(f"    [-] Error: {run['image_id']} {run['prime_id']} T={run['temperature']} S={run['seed']}: {e}")
                error_result = {
                    'image_id': run['image_id'],
                    'prime_id': run['prime_id'],
                    'temperature': run['temperature'],
                    'seed': run['seed'],
                    'error': str(e),
                    'timestamp': time.time()
                }
                f.write(json.dumps(error_result) + "\n")
                f.flush()
                errors += 1

    print(f"\n[+] Experiment complete!")
    print(f"    Output: {output_file}")
    print(f"    Total runs: {len(runs)}")
    print(f"    Errors: {errors}")


if __name__ == "__main__":
    main()
