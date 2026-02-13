import json
import pandas as pd
import itertools
from collections import defaultdict

# Load data
data = []
with open("semantic_vision_pilot_results.jsonl", "r") as f:
    for line in f:
        try:
            entry = json.loads(line)
            # Parse the inner JSON string from the model output
            raw = entry['raw_output'].replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            
            # Extract just the affordance descriptions
            affordances = " ".join([obj.get('affordance', '') for obj in parsed.get('objects', [])]).lower()
            
            data.append({
                'image': entry['image_id'],
                'prime': entry['prime_id'],
                'text': affordances
            })
        except:
            continue

df = pd.DataFrame(data)

# Calculate pairwise similarity for each image
similarities = []

def jaccard(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    if not a or not b: return 0.0
    return len(a.intersection(b)) / len(a.union(b))

print(f"Analyzing {len(df)} entries...")

for img in df['image'].unique():
    img_data = df[df['image'] == img]
    if len(img_data) < 2: continue
    
    pairs = itertools.combinations(img_data['prime'].unique(), 2)
    
    for p1, p2 in pairs:
        text1 = img_data[img_data['prime'] == p1]['text'].values[0]
        text2 = img_data[img_data['prime'] == p2]['text'].values[0]
        
        sim = jaccard(text1, text2)
        similarities.append({
            'pair': f"{p1} vs {p2}",
            'similarity': sim
        })

# Aggregate results
sim_df = pd.DataFrame(similarities)
summary = sim_df.groupby('pair')['similarity'].mean().sort_values()

print("\n=== PRELIMINARY RESULTS: CONTEXT DIVERGENCE ===")
print("Metric: Jaccard Similarity (Lower = Stronger Context Effect)")
print("-" * 50)
print(summary.head(10))
print("-" * 50)
print(f"Overall Mean Similarity: {sim_df['similarity'].mean():.4f}")
