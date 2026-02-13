import json
import os
from pathlib import Path

# Fix HF Cache Permissions
os.environ["HF_HOME"] = str(Path.home() / "Resurrexi/projects/data/hf_cache")

import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import tucker, parafac
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from collections import defaultdict

# Configuration
INPUT_FILE = "semantic_vision_pilot_results_qwen3_30b.jsonl"
EMBED_MODEL = "all-MiniLM-L6-v2"
RANK = [10, 3, 10] # [Image, Prime, Embedding] dimensions for Tucker

def load_data(filepath):
    print(f"[*] Loading data from {filepath}...")
    data = []
    with open(filepath, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if "error" in entry: continue
                
                # Extract affordance text
                raw = entry['raw_output'].replace("```json", "").replace("```", "").strip()
                parsed = json.loads(raw)
                
                # Combine all affordance descriptions into one text block
                affordances = " ".join([
                    f"{obj.get('name', '')} {obj.get('affordance', '')} {obj.get('reasoning', '')}" 
                    for obj in parsed.get('objects', [])
                ]).lower()
                
                data.append({
                    'image': entry['image_id'],
                    'prime': entry['prime_id'],
                    'text': affordances
                })
            except Exception as e:
                continue
    
    df = pd.DataFrame(data)
    print(f"[+] Loaded {len(df)} valid entries.")
    return df

def analyze_drift(df):
    print("\n[*] Analyzing Affordance Drift (Jaccard Similarity)...")
    similarities = []
    
    def jaccard(str1, str2):
        a = set(str1.split())
        b = set(str2.split())
        if not a or not b: return 0.0
        return len(a.intersection(b)) / len(a.union(b))

    for img in df['image'].unique():
        img_data = df[df['image'] == img]
        if len(img_data) < 2: continue
        
        pairs = itertools.combinations(img_data['prime'].unique(), 2)
        for p1, p2 in pairs:
            text1 = img_data[img_data['prime'] == p1]['text'].values[0]
            text2 = img_data[img_data['prime'] == p2]['text'].values[0]
            sim = jaccard(text1, text2)
            similarities.append({'pair': f"{p1} vs {p2}", 'similarity': sim})
            
    sim_df = pd.DataFrame(similarities)
    mean_sim = sim_df['similarity'].mean()
    print(f"[+] Mean Jaccard Similarity across all primes: {mean_sim:.4f}")
    print(f"    (Interpretation: {mean_sim*100:.1f}% overlap = {(1-mean_sim)*100:.1f}% context-driven shift)")
    return mean_sim

def tensor_analysis(df):
    print("\n[*] Performing Tensor Decomposition...")
    
    # 1. Embed text
    print("    > Generating embeddings...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(df['text'].tolist())
    df['embedding'] = list(embeddings)
    
    # 2. Construct Tensor (Images x Primes x Embed_Dim)
    images = sorted(df['image'].unique())
    primes = sorted(df['prime'].unique())
    embed_dim = embeddings.shape[1]
    
    # Filter to complete sets only
    valid_images = []
    for img in images:
        if len(df[df['image'] == img]) == len(primes):
            valid_images.append(img)
            
    print(f"    > Found {len(valid_images)} complete image sets for tensor construction.")
    if len(valid_images) == 0:
        print("[-] Insufficient data for full tensor.")
        return

    tensor_data = np.zeros((len(valid_images), len(primes), embed_dim))
    
    for i, img in enumerate(valid_images):
        for j, prime in enumerate(primes):
            subset = df[(df['image'] == img) & (df['prime'] == prime)]
            tensor_data[i, j, :] = subset['embedding'].values[0]
            
    # 3. Tucker Decomposition
    print(f"    > Running Tucker Decomposition (Rank {RANK})...")
    core, factors = tucker(tensor_data, rank=RANK, init='random', random_state=42)
    
    # Factor 1: Prime Latents
    prime_factors = factors[1] # Shape (7, 3)
    
    print("\n[+] Prime Factor Loadings (Latent Context Dimensions):")
    p_df = pd.DataFrame(prime_factors, index=primes, columns=[f"Dim_{i+1}" for i in range(RANK[1])])
    print(p_df)
    
    # Explained Variance
    rec = tl.tucker_to_tensor((core, factors))
    error = tl.norm(tensor_data - rec) / tl.norm(tensor_data)
    print(f"\n[+] Reconstruction Error: {error:.4f} (Explained Variance: {(1-error)*100:.1f}%)")

if __name__ == "__main__":
    df = load_data(INPUT_FILE)
    analyze_drift(df)
    tensor_analysis(df)
