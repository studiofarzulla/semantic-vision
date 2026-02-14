# Context-Dependent Affordance Computation in Vision-Language Models

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18091089-blue.svg)](https://doi.org/10.5281/zenodo.18091089)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**Working Paper DAI-2505**

Murad Farzulla — [Dissensus AI](https://dissensus.ai) / King's College London

**DOI:** [10.5281/zenodo.18091089](https://doi.org/10.5281/zenodo.18091089)

## Abstract

We characterize the phenomenon of *context-dependent affordance computation* in vision-language models (VLMs). Through a large-scale computational study (n=3,213 scene-context pairs from COCO-2017) using Qwen-VL 30B and LLaVA-1.5-13B subject to systematic context priming across 7 agentic personas, we demonstrate massive affordance drift: mean Jaccard similarity between context conditions is 0.095 (95% CI: [0.093, 0.096], p < 0.0001), indicating that >90% of lexical scene description is context-dependent. Sentence-level cosine similarity confirms substantial drift at the semantic level (mean = 0.415, 58.5% context-dependent). Stochastic baseline experiments confirm this drift reflects genuine context effects rather than generation noise. Tucker decomposition with bootstrap stability analysis reveals stable orthogonal latent factors. These findings establish that VLMs compute affordances in a substantially context-dependent manner and suggest a direction for robotics research: dynamic, query-dependent ontological projection (JIT Ontology) rather than static world modeling.

## Repository Structure

```
semantic-vision/
├── paper/
│   ├── semantic-vision-v2.tex        # LaTeX source (v2, canonical)
│   ├── semantic-vision-v2.pdf        # Compiled paper
│   ├── references.bib                # Bibliography (39 entries)
│   └── semantic-vision-v2-arxiv.tar.gz  # arXiv submission bundle
├── code/
│   ├── run_pilot.py                  # Pilot experiment runner
│   ├── analyze_full.py               # Full analysis pipeline
│   ├── analyze_prelim.py             # Preliminary analysis
│   ├── analyze_with_significance.py  # Statistical significance testing
│   ├── analyze_stochastic_baseline.py # Stochastic baseline analysis
│   ├── improved_similarity.py        # Alternative similarity metrics
│   ├── tucker_stability.py           # Tucker decomposition + bootstrap stability
│   ├── stochastic_baseline.py        # Stochastic baseline runner (model 1)
│   ├── stochastic_baseline_model2.py # Stochastic baseline runner (model 2)
│   ├── run_cross_model_replication.py # Cross-model replication (LLaVA)
│   ├── run_cross_model_ollama.py     # Cross-model replication via Ollama
│   ├── analysis/                     # Generated analysis outputs (figures, tables, JSON)
│   │   ├── improved_similarity/      # Alternative metric comparisons
│   │   ├── stochastic_baseline/      # Stochastic baseline figures
│   │   └── tucker_stability/         # Tucker decomposition figures
│   └── results/                      # Raw experimental results (JSONL)
└── CITATION.cff
```

## Code Description

The `code/` directory contains the full experimental pipeline:

- **Core experiment:** `run_pilot.py` runs VLM inference across 7 agentic personas (chef, child, architect, etc.) on COCO-2017 images, collecting scene descriptions under each context prime.
- **Statistical analysis:** `analyze_with_significance.py` computes Jaccard and cosine similarity across conditions with bootstrap confidence intervals and permutation tests.
- **Stochastic baselines:** `stochastic_baseline.py` and `stochastic_baseline_model2.py` run 2,384 inference passes across 4 temperatures and 5 seeds to separate context effects from generation noise.
- **Tucker decomposition:** `tucker_stability.py` performs tensor decomposition on the image-context-feature array with 1,000 bootstrap resamples for stability analysis.
- **Alternative metrics:** `improved_similarity.py` tests robustness across multiple similarity measures (Jaccard, cosine, soft cosine, BERTScore).
- **Cross-model replication:** `run_cross_model_replication.py` replicates the main experiment on LLaVA-1.5-13B via Ollama.

**Note:** The COCO-2017 validation images (~787MB) are not included. Download from [cocodataset.org](https://cocodataset.org/) and place in `code/data/coco/val2017/`.

## Citation

```bibtex
@article{farzulla2025affordance,
  author    = {Farzulla, Murad},
  title     = {Context-Dependent Affordance Computation in Vision-Language Models},
  year      = {2025},
  doi       = {10.5281/zenodo.18091089},
  url       = {https://doi.org/10.5281/zenodo.18091089}
}
```

## Zenodo

The paper is archived on Zenodo: [10.5281/zenodo.18091089](https://doi.org/10.5281/zenodo.18091089)

## License

- **Paper:** CC-BY-4.0
- **Code:** MIT

## Links

- Paper (Zenodo): https://doi.org/10.5281/zenodo.18091089
- Code (GitHub): https://github.com/studiofarzulla/semantic-vision
- ASCRI programme page: https://systems.ac/5/DAI-2505
- Dissensus AI: https://dissensus.ai
