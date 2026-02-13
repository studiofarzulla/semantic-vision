#!/usr/bin/env python3
"""
Semantic Vision - Full Statistical Analysis with Significance Tests

Computes:
1. Jaccard similarity statistics (corrected)
2. Tucker decomposition with factor loadings
3. Permutation tests for H-S1 (Jaccard < 0.5 significance)
4. Bootstrap confidence intervals for all statistics
5. LaTeX-ready tables

Author: Statistical analysis pipeline
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import itertools
import warnings
warnings.filterwarnings('ignore')

# Fix HF Cache before imports
os.environ["HF_HOME"] = str(Path.home() / "Resurrexi/projects/data/hf_cache")

import numpy as np
import pandas as pd
from scipy import stats
import tensorly as tl
from tensorly.decomposition import tucker
from sentence_transformers import SentenceTransformer

# Configuration
INPUT_FILE = Path(__file__).parent.parent / "semantic_vision_pilot_results_qwen3_30b.jsonl"
OUTPUT_JSON = Path(__file__).parent / "analysis_results.json"
OUTPUT_TEX = Path(__file__).parent / "results_tables.tex"
EMBED_MODEL = "all-MiniLM-L6-v2"
TUCKER_RANK = [10, 3, 10]  # [Image, Prime, Embedding] dimensions
N_BOOTSTRAP = 10000
N_PERMUTATIONS = 10000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def load_data(filepath):
    """Load and parse JSONL data."""
    print(f"[*] Loading data from {filepath}...")
    data = []
    errors = 0

    with open(filepath, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if "error" in entry:
                    errors += 1
                    continue

                # Extract affordance text
                raw = entry['raw_output'].replace("```json", "").replace("```", "").strip()
                parsed = json.loads(raw)

                # Combine all affordance descriptions
                affordances = " ".join([
                    f"{obj.get('name', '')} {obj.get('affordance', '')} {obj.get('reasoning', '')}"
                    for obj in parsed.get('objects', [])
                ]).lower()

                # Extract object names only (for Jaccard on object sets)
                object_names = set([
                    obj.get('name', '').lower().strip()
                    for obj in parsed.get('objects', [])
                    if obj.get('name', '').strip()
                ])

                data.append({
                    'image': entry['image_id'],
                    'prime': entry['prime_id'],
                    'text': affordances,
                    'object_names': object_names,
                    'num_objects': len(parsed.get('objects', []))
                })
            except Exception as e:
                errors += 1
                continue

    df = pd.DataFrame(data)
    print(f"[+] Loaded {len(df)} valid entries ({errors} errors/skipped)")
    print(f"    Images: {df['image'].nunique()}")
    print(f"    Primes: {df['prime'].nunique()}")
    return df


def jaccard_words(str1, str2):
    """Word-level Jaccard similarity."""
    a = set(str1.split())
    b = set(str2.split())
    if not a or not b:
        return 0.0
    return len(a.intersection(b)) / len(a.union(b))


def jaccard_objects(set1, set2):
    """Object-level Jaccard similarity."""
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))


def analyze_jaccard(df):
    """
    Compute Jaccard similarities between prime pairs for each image.
    Returns both word-level and object-level Jaccard.
    """
    print("\n[*] Analyzing Jaccard Similarity...")

    word_similarities = []
    object_similarities = []
    all_pairs = []

    for img in df['image'].unique():
        img_data = df[df['image'] == img]
        if len(img_data) < 2:
            continue

        pairs = list(itertools.combinations(img_data['prime'].unique(), 2))
        for p1, p2 in pairs:
            row1 = img_data[img_data['prime'] == p1].iloc[0]
            row2 = img_data[img_data['prime'] == p2].iloc[0]

            word_sim = jaccard_words(row1['text'], row2['text'])
            obj_sim = jaccard_objects(row1['object_names'], row2['object_names'])

            word_similarities.append(word_sim)
            object_similarities.append(obj_sim)
            all_pairs.append({
                'image': img,
                'prime1': p1,
                'prime2': p2,
                'jaccard_words': word_sim,
                'jaccard_objects': obj_sim
            })

    word_arr = np.array(word_similarities)
    obj_arr = np.array(object_similarities)

    results = {
        'word_level': {
            'mean': float(word_arr.mean()),
            'std': float(word_arr.std()),
            'median': float(np.median(word_arr)),
            'min': float(word_arr.min()),
            'max': float(word_arr.max()),
            'n_pairs': len(word_arr)
        },
        'object_level': {
            'mean': float(obj_arr.mean()),
            'std': float(obj_arr.std()),
            'median': float(np.median(obj_arr)),
            'min': float(obj_arr.min()),
            'max': float(obj_arr.max()),
            'n_pairs': len(obj_arr)
        }
    }

    print(f"[+] Word-level Jaccard: mean={results['word_level']['mean']:.4f} (SD={results['word_level']['std']:.4f})")
    print(f"[+] Object-level Jaccard: mean={results['object_level']['mean']:.4f} (SD={results['object_level']['std']:.4f})")
    print(f"    Total pairs analyzed: {len(word_arr)}")

    return results, word_arr, obj_arr, pd.DataFrame(all_pairs)


def bootstrap_ci(data, stat_func=np.mean, n_bootstrap=N_BOOTSTRAP, alpha=0.05):
    """Compute bootstrap confidence interval."""
    n = len(data)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(stat_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    se = np.std(bootstrap_stats)

    return lower, upper, se


def permutation_test_less_than(data, threshold=0.5, n_perm=N_PERMUTATIONS):
    """
    Permutation test: Is the mean significantly less than threshold?
    H0: mean >= threshold
    H1: mean < threshold
    """
    observed_mean = np.mean(data)
    observed_diff = observed_mean - threshold

    # Under H0, we shift data so mean = threshold
    shifted = data + (threshold - observed_mean)

    count_more_extreme = 0
    for _ in range(n_perm):
        # Resample from shifted distribution
        perm_sample = np.random.choice(shifted, size=len(data), replace=True)
        perm_mean = np.mean(perm_sample)
        # Count how often permuted mean is as extreme or more extreme
        if perm_mean <= observed_mean:
            count_more_extreme += 1

    p_value = count_more_extreme / n_perm
    return p_value, observed_mean, observed_diff


def significance_tests(word_arr, obj_arr):
    """Run all significance tests."""
    print("\n[*] Running Significance Tests...")

    results = {}

    # Test H-S1: Is Jaccard < 0.5 (indicating significant context-driven variation)?
    print("  > Testing H-S1: Jaccard < 0.5 (word-level)...")
    p_word, mean_word, diff_word = permutation_test_less_than(word_arr, 0.5, N_PERMUTATIONS)

    print("  > Testing H-S1: Jaccard < 0.5 (object-level)...")
    p_obj, mean_obj, diff_obj = permutation_test_less_than(obj_arr, 0.5, N_PERMUTATIONS)

    # Also do parametric t-test for comparison
    t_word, p_t_word = stats.ttest_1samp(word_arr, 0.5)
    t_obj, p_t_obj = stats.ttest_1samp(obj_arr, 0.5)
    # One-sided: divide p by 2 for left-tail
    p_t_word_one = p_t_word / 2 if t_word < 0 else 1 - p_t_word / 2
    p_t_obj_one = p_t_obj / 2 if t_obj < 0 else 1 - p_t_obj / 2

    results['h_s1_word'] = {
        'test': 'Permutation test: mean < 0.5',
        'observed_mean': float(mean_word),
        'difference_from_threshold': float(diff_word),
        'p_value_permutation': float(p_word),
        'p_value_ttest_onesided': float(p_t_word_one),
        't_statistic': float(t_word),
        'significant_at_001': p_word < 0.01
    }

    results['h_s1_object'] = {
        'test': 'Permutation test: mean < 0.5',
        'observed_mean': float(mean_obj),
        'difference_from_threshold': float(diff_obj),
        'p_value_permutation': float(p_obj),
        'p_value_ttest_onesided': float(p_t_obj_one),
        't_statistic': float(t_obj),
        'significant_at_001': p_obj < 0.01
    }

    # Bootstrap CIs
    print("  > Computing bootstrap CIs...")
    lower_word, upper_word, se_word = bootstrap_ci(word_arr)
    lower_obj, upper_obj, se_obj = bootstrap_ci(obj_arr)

    results['bootstrap_ci_word'] = {
        'mean': float(np.mean(word_arr)),
        'ci_lower': float(lower_word),
        'ci_upper': float(upper_word),
        'se': float(se_word),
        'n_bootstrap': N_BOOTSTRAP
    }

    results['bootstrap_ci_object'] = {
        'mean': float(np.mean(obj_arr)),
        'ci_lower': float(lower_obj),
        'ci_upper': float(upper_obj),
        'se': float(se_obj),
        'n_bootstrap': N_BOOTSTRAP
    }

    print(f"[+] Word-level: mean={mean_word:.4f}, 95% CI=[{lower_word:.4f}, {upper_word:.4f}], p={p_word:.4f}")
    print(f"[+] Object-level: mean={mean_obj:.4f}, 95% CI=[{lower_obj:.4f}, {upper_obj:.4f}], p={p_obj:.4f}")

    return results


def tensor_analysis(df):
    """Perform Tucker decomposition on embeddings."""
    print("\n[*] Performing Tensor Decomposition...")

    # Embed text
    print("    > Generating embeddings...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
    df = df.copy()
    df['embedding'] = list(embeddings)

    # Construct Tensor (Images x Primes x Embed_Dim)
    images = sorted(df['image'].unique())
    primes = sorted(df['prime'].unique())
    embed_dim = embeddings.shape[1]

    # Filter to complete sets only
    valid_images = []
    for img in images:
        if len(df[df['image'] == img]) == len(primes):
            valid_images.append(img)

    print(f"    > Found {len(valid_images)} complete image sets (out of {len(images)} total)")

    if len(valid_images) == 0:
        print("[-] Insufficient data for tensor construction")
        return None

    tensor_data = np.zeros((len(valid_images), len(primes), embed_dim))

    for i, img in enumerate(valid_images):
        for j, prime in enumerate(primes):
            subset = df[(df['image'] == img) & (df['prime'] == prime)]
            if len(subset) > 0:
                tensor_data[i, j, :] = subset['embedding'].values[0]

    # Tucker Decomposition
    print(f"    > Running Tucker Decomposition (Rank {TUCKER_RANK})...")
    core, factors = tucker(tensor_data, rank=TUCKER_RANK, init='random', random_state=RANDOM_SEED)

    # Factor 1: Prime Latents
    prime_factors = factors[1]  # Shape (7, 3)

    # Explained Variance
    rec = tl.tucker_to_tensor((core, factors))
    total_var = np.sum(tensor_data ** 2)
    residual_var = np.sum((tensor_data - rec) ** 2)
    explained_var = 1 - (residual_var / total_var)
    reconstruction_error = np.linalg.norm(tensor_data - rec) / np.linalg.norm(tensor_data)

    # Factor loadings analysis
    prime_df = pd.DataFrame(
        prime_factors,
        index=primes,
        columns=[f"Dim_{i+1}" for i in range(TUCKER_RANK[1])]
    )

    # Variance explained by each dimension
    dim_variances = np.var(prime_factors, axis=0)
    dim_var_pct = dim_variances / dim_variances.sum() * 100

    results = {
        'tensor_shape': list(tensor_data.shape),
        'tucker_rank': TUCKER_RANK,
        'n_complete_images': len(valid_images),
        'explained_variance': float(explained_var),
        'reconstruction_error': float(reconstruction_error),
        'prime_factor_loadings': prime_df.to_dict(),
        'dimension_variance_pct': {f"Dim_{i+1}": float(v) for i, v in enumerate(dim_var_pct)},
        'primes': primes
    }

    print(f"\n[+] Tucker Decomposition Results:")
    print(f"    Explained Variance: {explained_var*100:.1f}%")
    print(f"    Reconstruction Error: {reconstruction_error:.4f}")
    print(f"\n[+] Prime Factor Loadings:")
    print(prime_df.round(4).to_string())
    print(f"\n[+] Dimension Variance Distribution:")
    for i, v in enumerate(dim_var_pct):
        print(f"    Dim_{i+1}: {v:.1f}%")

    return results, prime_factors, primes


def bootstrap_tucker_factors(df, n_bootstrap=100):
    """Bootstrap confidence intervals for Tucker factor loadings."""
    print(f"\n[*] Bootstrapping Tucker factors ({n_bootstrap} iterations)...")

    # Embed text once
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=False)
    df = df.copy()
    df['embedding'] = list(embeddings)

    images = sorted(df['image'].unique())
    primes = sorted(df['prime'].unique())
    embed_dim = embeddings.shape[1]

    # Filter to complete sets
    valid_images = [img for img in images if len(df[df['image'] == img]) == len(primes)]

    if len(valid_images) < 10:
        print("[-] Not enough complete images for bootstrap")
        return None

    all_prime_factors = []

    for b in range(n_bootstrap):
        if b % 20 == 0:
            print(f"    > Bootstrap iteration {b}/{n_bootstrap}")

        # Sample images with replacement
        sampled_images = np.random.choice(valid_images, size=len(valid_images), replace=True)

        # Build tensor
        tensor_data = np.zeros((len(sampled_images), len(primes), embed_dim))
        for i, img in enumerate(sampled_images):
            for j, prime in enumerate(primes):
                subset = df[(df['image'] == img) & (df['prime'] == prime)]
                if len(subset) > 0:
                    tensor_data[i, j, :] = subset['embedding'].values[0]

        try:
            core, factors = tucker(tensor_data, rank=TUCKER_RANK, init='random', random_state=b)
            all_prime_factors.append(factors[1])
        except:
            continue

    all_prime_factors = np.array(all_prime_factors)  # (n_bootstrap, n_primes, n_dims)

    # Compute CIs for each factor loading
    ci_results = {}
    for j, prime in enumerate(primes):
        ci_results[prime] = {}
        for d in range(TUCKER_RANK[1]):
            values = all_prime_factors[:, j, d]
            ci_results[prime][f"Dim_{d+1}"] = {
                'mean': float(np.mean(values)),
                'ci_lower': float(np.percentile(values, 2.5)),
                'ci_upper': float(np.percentile(values, 97.5)),
                'se': float(np.std(values))
            }

    print(f"[+] Bootstrap CIs computed for {len(primes)} primes x {TUCKER_RANK[1]} dimensions")
    return ci_results


def generate_latex_tables(results):
    """Generate LaTeX tables for paper."""
    latex = []

    # Table 1: Jaccard Similarity Summary
    latex.append(r"""
%==============================================================================
% Table: Jaccard Similarity Statistics
%==============================================================================
\begin{table}[H]
\centering
\caption{Jaccard Similarity Between Context Primes}
\label{tab:jaccard-stats}
\small
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Metric} & \textbf{Mean} & \textbf{SD} & \textbf{95\% CI} & \textbf{$t$} & \textbf{$p$} & \textbf{$n$} \\
\midrule""")

    js = results['jaccard_stats']
    sig = results['significance_tests']

    # Word-level
    w = js['word_level']
    sw = sig['bootstrap_ci_word']
    tw = sig['h_s1_word']
    latex.append(f"Word-level & {w['mean']:.4f} & {w['std']:.4f} & [{sw['ci_lower']:.4f}, {sw['ci_upper']:.4f}] & {tw['t_statistic']:.2f} & {tw['p_value_permutation']:.4f} & {w['n_pairs']} \\\\")

    # Object-level
    o = js['object_level']
    so = sig['bootstrap_ci_object']
    to = sig['h_s1_object']
    latex.append(f"Object-level & {o['mean']:.4f} & {o['std']:.4f} & [{so['ci_lower']:.4f}, {so['ci_upper']:.4f}] & {to['t_statistic']:.2f} & {to['p_value_permutation']:.4f} & {o['n_pairs']} \\\\")

    latex.append(r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: $p$-values from permutation test (10,000 iterations) for H$_0$: $\mu \geq 0.5$.
\item 95\% CIs from bootstrap (10,000 resamples).
\end{tablenotes}
\end{table}
""")

    # Table 2: Tucker Decomposition
    if 'tucker_results' in results and results['tucker_results']:
        tr = results['tucker_results']
        latex.append(r"""
%==============================================================================
% Table: Tucker Decomposition Results
%==============================================================================
\begin{table}[H]
\centering
\caption{Tucker Decomposition: Context Prime Factor Loadings}
\label{tab:tucker-factors}
\small
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Prime} & \textbf{Dim$_1$} & \textbf{Dim$_2$} & \textbf{Dim$_3$} \\
\midrule""")

        loadings = tr['prime_factor_loadings']
        for prime in tr['primes']:
            d1 = loadings['Dim_1'][prime]
            d2 = loadings['Dim_2'][prime]
            d3 = loadings['Dim_3'][prime]
            # Clean prime name for display
            prime_clean = prime.replace('_', ' ').replace('P0', 'P0:').replace('P1', 'P1:').replace('P2', 'P2:').replace('P3', 'P3:').replace('P4', 'P4:').replace('P5', 'P5:').replace('P6', 'P6:')
            latex.append(f"{prime_clean} & {d1:.4f} & {d2:.4f} & {d3:.4f} \\\\")

        latex.append(r"""\midrule
\textbf{Var. \%} & """ + " & ".join([f"{tr['dimension_variance_pct'][f'Dim_{i+1}']:.1f}\\%" for i in range(3)]) + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Explained variance: """ + f"{tr['explained_variance']*100:.1f}" + r"""\%. Tensor shape: """ + f"{tr['tensor_shape'][0]} images $\\times$ {tr['tensor_shape'][1]} primes $\\times$ {tr['tensor_shape'][2]} embedding dims" + r""".
\end{tablenotes}
\end{table}
""")

    # Table 3: Hypothesis Test Summary
    latex.append(r"""
%==============================================================================
% Table: Hypothesis Test Results
%==============================================================================
\begin{table}[H]
\centering
\caption{Hypothesis Test Results Summary}
\label{tab:hypothesis-tests}
\small
\begin{tabular}{@{}lllcc@{}}
\toprule
\textbf{Hypothesis} & \textbf{Test} & \textbf{Result} & \textbf{$p$-value} & \textbf{Decision} \\
\midrule""")

    # H-S1 word
    hw = sig['h_s1_word']
    decision_w = "Supported" if hw['significant_at_001'] else "Not Supported"
    latex.append(f"H-S1 (word) & $J_{{\\text{{word}}}} < 0.5$ & $\\bar{{J}} = {hw['observed_mean']:.4f}$ & {hw['p_value_permutation']:.4f} & {decision_w} \\\\")

    # H-S1 object
    ho = sig['h_s1_object']
    decision_o = "Supported" if ho['significant_at_001'] else "Not Supported"
    latex.append(f"H-S1 (object) & $J_{{\\text{{obj}}}} < 0.5$ & $\\bar{{J}} = {ho['observed_mean']:.4f}$ & {ho['p_value_permutation']:.4f} & {decision_o} \\\\")

    # H-V1 (Tucker)
    if 'tucker_results' in results and results['tucker_results']:
        tr = results['tucker_results']
        h_v1_result = "Supported" if tr['explained_variance'] > 0.7 else "Not Supported"
        latex.append(f"H-V1 & Explained Var. $> 70\\%$ & ${tr['explained_variance']*100:.1f}\\%$ & --- & {h_v1_result} \\\\")

    latex.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

    return "\n".join(latex)


def main():
    print("=" * 70)
    print("SEMANTIC VISION - STATISTICAL ANALYSIS WITH SIGNIFICANCE TESTS")
    print("=" * 70)

    # Load data
    df = load_data(INPUT_FILE)

    # Jaccard analysis
    jaccard_stats, word_arr, obj_arr, pairs_df = analyze_jaccard(df)

    # Significance tests
    sig_results = significance_tests(word_arr, obj_arr)

    # Tucker decomposition
    tucker_result = tensor_analysis(df)
    tucker_stats = tucker_result[0] if tucker_result else None

    # Bootstrap Tucker CIs (reduced iterations for speed)
    tucker_bootstrap = bootstrap_tucker_factors(df, n_bootstrap=100)

    # Compile all results
    all_results = {
        'metadata': {
            'input_file': str(INPUT_FILE),
            'n_entries': len(df),
            'n_images': int(df['image'].nunique()),
            'n_primes': int(df['prime'].nunique()),
            'n_bootstrap': N_BOOTSTRAP,
            'n_permutations': N_PERMUTATIONS,
            'random_seed': RANDOM_SEED
        },
        'jaccard_stats': jaccard_stats,
        'significance_tests': sig_results,
        'tucker_results': tucker_stats,
        'tucker_bootstrap_ci': tucker_bootstrap
    }

    # Save JSON results
    print(f"\n[*] Saving results to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate LaTeX tables
    latex_content = generate_latex_tables(all_results)
    print(f"[*] Saving LaTeX tables to {OUTPUT_TEX}...")
    with open(OUTPUT_TEX, 'w') as f:
        f.write(latex_content)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n[JACCARD SIMILARITY]")
    print(f"  Word-level:   mean = {jaccard_stats['word_level']['mean']:.4f}, SD = {jaccard_stats['word_level']['std']:.4f}")
    print(f"                95% CI = [{sig_results['bootstrap_ci_word']['ci_lower']:.4f}, {sig_results['bootstrap_ci_word']['ci_upper']:.4f}]")
    print(f"  Object-level: mean = {jaccard_stats['object_level']['mean']:.4f}, SD = {jaccard_stats['object_level']['std']:.4f}")
    print(f"                95% CI = [{sig_results['bootstrap_ci_object']['ci_lower']:.4f}, {sig_results['bootstrap_ci_object']['ci_upper']:.4f}]")

    print(f"\n[SIGNIFICANCE TESTS]")
    print(f"  H-S1 (word):   p = {sig_results['h_s1_word']['p_value_permutation']:.4f} {'***' if sig_results['h_s1_word']['p_value_permutation'] < 0.001 else '**' if sig_results['h_s1_word']['p_value_permutation'] < 0.01 else '*' if sig_results['h_s1_word']['p_value_permutation'] < 0.05 else ''}")
    print(f"  H-S1 (object): p = {sig_results['h_s1_object']['p_value_permutation']:.4f} {'***' if sig_results['h_s1_object']['p_value_permutation'] < 0.001 else '**' if sig_results['h_s1_object']['p_value_permutation'] < 0.01 else '*' if sig_results['h_s1_object']['p_value_permutation'] < 0.05 else ''}")

    if tucker_stats:
        print(f"\n[TUCKER DECOMPOSITION]")
        print(f"  Explained variance: {tucker_stats['explained_variance']*100:.1f}%")
        print(f"  Complete images used: {tucker_stats['n_complete_images']}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"  Results: {OUTPUT_JSON}")
    print(f"  LaTeX:   {OUTPUT_TEX}")
    print("=" * 70)


if __name__ == "__main__":
    main()
