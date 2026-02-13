import base64
import json
import os
import time
import argparse
from pathlib import Path
from openai import OpenAI

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Point this to your COCO val2017 folder
DEFAULT_IMAGE_DIR = str(Path.home() / "Resurrexi/data/coco/val2017")
OUTPUT_FILE = "semantic_vision_pilot_results.jsonl"
API_BASE = "http://192.168.2.21:1234/v1"
API_KEY = "lm-studio" # Not used locally but required by client
MODEL_NAME = "unsloth/qwen3-vl-30b-a3b-instruct" # Placeholder, actual name handled by server usually

# The 7 Context Primes from the Semantic-First Paper
PRIMES = {
    "P0_Neutral": "Analyze this image objectively. List the 3 most prominent objects, their geometric properties, and standard functions.",
    
    "P1_Chef": "You are a professional chef examining this scene for cooking-related possibilities. Identify the 3 most critical items for food preparation and list their affordances (what you can do with them).",
    
    "P2_Security": "You are a security professional assessing this space for vulnerabilities and tactical assets. Identify 3 objects that represent risks or defensive tools and their affordances.",
    
    "P3_Child": "Imagine you are a 4-year-old child. Identify 3 interesting things to play with in this scene and how you would use them.",
    
    "P4_Mobility": "You are navigating this space in a wheelchair. Identify 3 objects that either obstruct your path or enable your movement.",
    
    "P5_Urgent": "EMERGENCY: You have 30 seconds to find a tool for immediate survival. What do you see first and how do you use it?",
    
    "P6_Leisure": "You are casually exploring this space with absolutely no time pressure. What catches your eye for pure enjoyment or relaxation?"
}

# ==============================================================================
# UTILITIES
# ==============================================================================

def encode_image(image_path):
    """Encodes image to base64 for API transmission."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_args():
    parser = argparse.ArgumentParser(description="Run Semantic-First Vision Pilot")
    parser.add_argument("--images", type=str, default=DEFAULT_IMAGE_DIR, help="Path to image directory")
    parser.add_argument("--limit", type=int, default=500, help="Max images to process")
    parser.add_argument("--resume", action="store_true", help="Resume from last processed image")
    return parser.parse_args()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def run_pilot():
    args = parse_args()
    
    # Setup Client
    print(f"[*] Connecting to local inference server at {API_BASE}...")
    try:
        client = OpenAI(base_url=API_BASE, api_key=API_KEY)
        # Test connection (optional, just listing models)
        client.models.list()
        print("[+] Connection successful.")
    except Exception as e:
        print(f"[-] Failed to connect: {e}")
        return

    # Image Discovery
    img_dir = Path(args.images)
    if not img_dir.exists():
        print(f"[-] Image directory not found: {img_dir}")
        print("    Please create it or specify --images path")
        return

    images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    if not images:
        print("[-] No images found.")
        return
        
    print(f"[*] Found {len(images)} images. Processing first {args.limit}...")
    images = images[:args.limit]

    # Resume Logic
    processed_count = 0
    if args.resume and os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            processed_count = sum(1 for _ in f)
        # Each image generates len(PRIMES) entries
        images_done = processed_count // len(PRIMES)
        images = images[images_done:]
        print(f"[*] Resuming from image index {images_done}...")

    # Processing Loop
    with open(OUTPUT_FILE, "a") as f:
        for i, img_name in enumerate(images):
            img_path = img_dir / img_name
            try:
                base64_image = encode_image(img_path)
            except Exception as e:
                print(f"[-] Failed to read {img_name}: {e}")
                continue
            
            print(f"\n[{i+1}/{len(images)}] Processing {img_name}...")
            
            for p_id, prime_text in PRIMES.items():
                start_time = time.time()
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME, 
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": f"{prime_text}\n\nProvide response in JSON format with keys: 'objects' (list of {{id, name, affordance, reasoning}})."},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                ]
                            }
                        ],
                        max_tokens=512,
                        temperature=0.7, # Slight creativity for divergent affordances
                    )
                    
                    output_content = response.choices[0].message.content
                    
                    result = {
                        "image_id": img_name,
                        "prime_id": p_id,
                        "prime_text": prime_text,
                        "raw_output": output_content,
                        "timestamp": time.time(),
                        "inference_time": time.time() - start_time
                    }
                    
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    print(f"    > {p_id}: Success ({result['inference_time']:.2f}s)")
                    
                except Exception as e:
                    print(f"    > {p_id}: FAILED ({e})")
                    # Log error but continue
                    error_result = {
                        "image_id": img_name,
                        "prime_id": p_id,
                        "error": str(e),
                        "timestamp": time.time()
                    }
                    f.write(json.dumps(error_result) + "\n")
                    f.flush()

if __name__ == "__main__":
    run_pilot()
