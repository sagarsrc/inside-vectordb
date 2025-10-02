# %% [markdown]
# # FAISS Demo: Fast Approximate Nearest Neighbor Search
#
# This notebook demonstrates **FAISS (Facebook AI Similarity Search)** with HNSW index.
#
# **What You'll Learn:**
# - How FAISS HNSW index works
# - Key parameters: M, efConstruction, efSearch
# - Performance comparison vs hnswlib
#
# **Why FAISS?**
# - Production-ready library from Meta
# - Multiple index types (HNSW, IVF, PQ, etc.)
# - Highly optimized C++ backend
# - Used by Meta, Uber, Spotify, and others

# %% [markdown]
# ## Configuration

# %% Global Configuration
import os

DATA_ROOT = "../data"
REPORTS_DIR = "../reports"
INDEX_DIR = "../data/index"
INDEX_PATH = "../data/index/faiss_index.bin"
DATASET_NAME = "msmarco"
USE_SUBSET = True
SUBSET_SIZE = "1M"
SPLIT = "dev"

# FAISS HNSW Parameters
M = 32  # Number of connections per layer (typical: 16-64)
EF_CONSTRUCTION = 100  # Build-time search width (reduced for faster build)
EF_SEARCH = 50  # Search-time width

# Benchmark settings
N_QUERY_SAMPLES = 100  # Number of queries to test
K_VALUES = [1, 5, 10, 20, 50, 100]  # K for recall@K evaluation

# %% [markdown]
# ## Import Dependencies

# %% Imports
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import faiss
import time
from pathlib import Path
import matplotlib.pyplot as plt

# Import shared utilities
from utils import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr,
    save_metrics_report,
)

try:
    print(f"FAISS version: {faiss.__version__}")
except AttributeError:
    print("FAISS imported successfully")

# Create reports directory structure
FAISS_REPORTS = f"{REPORTS_DIR}/faiss"
Path(FAISS_REPORTS).mkdir(parents=True, exist_ok=True)
print(f"Reports will be saved to: {FAISS_REPORTS}")


# %% [markdown]
# ## Data Loading Functions


# %% Data Loading
def load_embeddings(npz_path: str):
    """Load embeddings from NPZ file."""
    print(f"Loading: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    ids = data["ids"].tolist()
    print(f"  Loaded {len(ids):,} embeddings with dimension {embeddings.shape[1]}")
    return embeddings, ids


def load_qrels(data_root: str, dataset_name: str, split: str = "dev"):
    """Load ground truth relevance judgments."""
    dataset_path = f"{data_root}/{dataset_name}"

    # Load qrels
    qrels_file = f"{dataset_path}/qrels/{split}.tsv"
    qrels = {}
    with open(qrels_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            query_id, doc_id, score = line.strip().split("\t")
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = int(score)

    print(f"\nLoaded qrels for {len(qrels):,} queries")
    print(f"Total relevance judgments: {sum(len(docs) for docs in qrels.values()):,}")

    return qrels


def sample_queries(query_embeddings, query_ids, qrels, n_samples=None):
    """Sample queries that have ground truth."""
    query_indices = []
    query_id_list = []

    for i, qid in enumerate(query_ids):
        if qid in qrels:
            query_indices.append(i)
            query_id_list.append(qid)

    if n_samples and n_samples < len(query_indices):
        np.random.seed(42)
        sample_idx = np.random.choice(len(query_indices), n_samples, replace=False)
        query_indices = [query_indices[i] for i in sample_idx]
        query_id_list = [query_id_list[i] for i in sample_idx]

    query_emb = query_embeddings[query_indices]

    print(f"\nUsing {len(query_id_list):,} queries for evaluation")

    return query_emb, query_id_list


# %% Load Data
def load_all_data():
    """Load embeddings and qrels."""
    subset_suffix = f"_{SUBSET_SIZE}" if USE_SUBSET else ""
    corpus_emb_path = (
        f"{DATA_ROOT}/{DATASET_NAME}{subset_suffix}_corpus_embeddings.npz"
    )
    query_emb_path = f"{DATA_ROOT}/{DATASET_NAME}{subset_suffix}_query_embeddings.npz"

    corpus_embeddings, corpus_ids = load_embeddings(corpus_emb_path)
    query_embeddings, query_ids = load_embeddings(query_emb_path)
    qrels = load_qrels(DATA_ROOT, DATASET_NAME, split=SPLIT)

    return corpus_embeddings, corpus_ids, query_embeddings, query_ids, qrels


corpus_embeddings, corpus_ids, query_embeddings, query_ids, qrels = load_all_data()
query_emb, query_id_list = sample_queries(
    query_embeddings, query_ids, qrels, N_QUERY_SAMPLES
)


# %% [markdown]
# ## Build FAISS HNSW Index
#
# **Key Parameters:**
# - `M = 32`: Number of connections per node (FAISS default is higher than hnswlib)
# - `efConstruction = 100`: Build-time search width
#   - Reduced from 200 to 100 for **2x faster** build time with minimal quality loss
#   - For 1M docs: ~3-5 minutes instead of 10+ minutes
# - FAISS automatically normalizes vectors for cosine similarity
#
# **Build Optimizations:**
# - Batch insertion (50K vectors/batch) for better memory locality
# - Progress tracking to monitor build status
# - L2 normalization for accurate cosine similarity


# %% FAISS Index Building
def build_faiss_hnsw_index(embeddings, m, ef_construction):
    """Build FAISS HNSW index."""
    dim = embeddings.shape[1]
    n_elements = len(embeddings)

    print(f"\n{'=' * 80}")
    print(f"BUILDING FAISS HNSW INDEX")
    print(f"{'=' * 80}\n")

    print(f"Dataset: {DATASET_NAME} ({SUBSET_SIZE} subset)" if USE_SUBSET else DATASET_NAME)
    print(f"Corpus size: {n_elements:,} documents")
    print(f"Dimension: {dim}")
    print(f"\nFAISS HNSW Parameters:")
    print(f"  M (connections per layer): {m}")
    print(f"  efConstruction: {ef_construction}")
    print(f"  Metric: Inner Product (for normalized vectors = cosine)")

    # Create HNSW index
    index = faiss.IndexHNSWFlat(dim, m)
    index.hnsw.efConstruction = ef_construction

    # Normalize embeddings for cosine similarity
    print("\nNormalizing embeddings for cosine similarity...")
    embeddings_normalized = embeddings.copy()
    faiss.normalize_L2(embeddings_normalized)

    # Add items in batches with progress tracking
    batch_size = 1000
    start_time = time.time()

    print(f"\nBuilding index (batch size: {batch_size:,})...")
    for i in range(0, n_elements, batch_size):
        end_idx = min(i + batch_size, n_elements)
        batch = embeddings_normalized[i:end_idx]
        index.add(batch)

        # Progress update
        progress = (end_idx / n_elements) * 100
        elapsed = time.time() - start_time
        print(f"  Progress: {progress:5.1f}% ({end_idx:,}/{n_elements:,}) - {elapsed:.1f}s elapsed", end="\r")

    build_time = time.time() - start_time
    print()  # New line after progress
    print(f"\nIndex built in {build_time:.2f} seconds ({build_time/60:.1f} minutes)")
    print(f"Build speed: {n_elements / build_time:.0f} vectors/sec")
    print(f"\nIndex Statistics:")
    print(f"  Total elements: {index.ntotal:,}")

    return index, build_time, embeddings_normalized


# Load or build FAISS index
index_file = Path(INDEX_PATH)
if index_file.exists():
    print(f"\n{'=' * 80}")
    print(f"LOADING EXISTING FAISS INDEX")
    print(f"{'=' * 80}\n")
    print(f"Loading index from: {index_file}")

    index = faiss.read_index(str(index_file))

    print(f"Index loaded successfully")
    print(f"  Total elements: {index.ntotal:,}")

    # Normalize embeddings (needed for search later)
    corpus_embeddings_normalized = corpus_embeddings.copy()
    faiss.normalize_L2(corpus_embeddings_normalized)

    build_time = 0  # No build time since we loaded from disk
else:
    index, build_time, corpus_embeddings_normalized = build_faiss_hnsw_index(
        corpus_embeddings, M, EF_CONSTRUCTION
    )

    # Save index to disk
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_file))
    print(f"\nIndex saved to: {index_file}")

# %% [markdown]
# ## Analyze Search Behavior


# %% Search Behavior Analysis
def analyze_search_behavior(index, query_emb):
    """Analyze FAISS search behavior with different efSearch values."""
    # Normalize query embeddings
    query_emb_normalized = query_emb.copy()
    faiss.normalize_L2(query_emb_normalized)

    sample_queries = query_emb_normalized[:5]
    ef_test_values = [10, 50, 200]
    k_test = 10

    print(f"\n{'=' * 80}")
    print(f"ANALYZING FAISS HNSW INDEX BEHAVIOR")
    print(f"{'=' * 80}\n")

    print("Demonstrating FAISS HNSW search with different efSearch values:")
    print("(Shows how efSearch trades off speed vs accuracy)\n")

    for ef_val in ef_test_values:
        index.hnsw.efSearch = ef_val

        # Time search
        start = time.time()
        distances, labels = index.search(sample_queries, k_test)
        search_time = (time.time() - start) * 1000 / len(sample_queries)

        print(f"efSearch={ef_val:3d}: {search_time:.2f} ms/query")
        print(
            f"  Query 0 top-3 neighbors: {labels[0][:3]} (distances: {distances[0][:3].round(3)})"
        )

    print("\n** FAISS HNSW Properties **")
    print("- Lower efSearch → Faster search (explores fewer nodes)")
    print("- Higher efSearch → More accurate (explores more of the graph)")
    print("- FAISS uses inner product metric (equivalent to cosine after normalization)")


analyze_search_behavior(index, query_emb)


# %% [markdown]
# ## Search with FAISS


# %% FAISS Search
def perform_faiss_search(index, query_emb, query_id_list, corpus_ids, ef_search, k_max):
    """Perform FAISS HNSW search on all queries."""
    print(f"\n{'=' * 80}")
    print(f"FAISS HNSW SEARCH")
    print(f"{'=' * 80}\n")

    print(f"Search parameters:")
    print(f"  efSearch: {ef_search}")
    print(f"  K (neighbors to retrieve): {k_max}")

    # Set search parameter
    index.hnsw.efSearch = ef_search

    # Normalize query embeddings
    query_emb_normalized = query_emb.copy()
    faiss.normalize_L2(query_emb_normalized)

    # Perform search with timing
    print(f"\nSearching {len(query_emb):,} queries...")
    start_time = time.time()
    distances, labels = index.search(query_emb_normalized, k_max)
    search_time = time.time() - start_time

    # Convert to results dictionary
    all_results = {}
    for i, qid in enumerate(query_id_list):
        retrieved_indices = labels[i]
        retrieved_doc_ids = [corpus_ids[idx] for idx in retrieved_indices]
        all_results[qid] = retrieved_doc_ids

    print(f"Search completed in {search_time:.2f} seconds")
    print(f"Queries per second: {len(query_emb) / search_time:.1f}")
    print(f"Avg latency per query: {search_time * 1000 / len(query_emb):.2f} ms")

    return all_results, search_time


all_results, search_time = perform_faiss_search(
    index, query_emb, query_id_list, corpus_ids, EF_SEARCH, max(K_VALUES)
)


# %% [markdown]
# ## Evaluate Performance


# %% Calculate Metrics
def calculate_metrics(results, qrels, k_values):
    """Calculate recall, precision, and MRR."""
    # Convert dict format to list format expected by utils functions
    retrieved_ids = []
    relevant_ids = []

    for query_id in results.keys():
        retrieved_ids.append(results[query_id])
        # Get relevant doc IDs from qrels
        relevant_docs = list(qrels.get(query_id, {}).keys())
        relevant_ids.append(relevant_docs)

    # Compute metrics using utils functions
    recall_scores = compute_recall_at_k(retrieved_ids, relevant_ids, k_values=k_values)
    precision_scores = compute_precision_at_k(retrieved_ids, relevant_ids, k_values=[1, 5, 10])
    mrr_score = compute_mrr(retrieved_ids, relevant_ids)

    # Display results
    print(f"\n{'=' * 80}")
    print(f"FAISS HNSW PERFORMANCE METRICS")
    print(f"{'=' * 80}\n")

    print(f"Recall@K:")
    for k, score in recall_scores.items():
        print(f"  Recall@{k}: {score:.4f}")

    print(f"\nPrecision@K:")
    for k, score in precision_scores.items():
        print(f"  Precision@{k}: {score:.4f}")

    print(f"\nMRR: {mrr_score:.4f}")

    return recall_scores, precision_scores, mrr_score


recall_scores, precision_scores, mrr_score = calculate_metrics(
    all_results, qrels, K_VALUES
)


# %% [markdown]
# ## Parameter Sensitivity Analysis


# %% Parameter Analysis
def analyze_ef_parameter(index, query_emb, query_id_list, corpus_ids, qrels):
    """Analyze efSearch parameter sensitivity."""
    print(f"\n{'=' * 80}")
    print(f"PARAMETER SENSITIVITY: efSearch")
    print(f"{'=' * 80}\n")

    # Normalize queries
    query_emb_normalized = query_emb.copy()
    faiss.normalize_L2(query_emb_normalized)

    ef_values = [10, 20, 50, 100, 200, 500]
    ef_results = []

    for ef in ef_values:
        index.hnsw.efSearch = ef

        # Time search
        start = time.time()
        distances, labels = index.search(query_emb_normalized, 10)
        search_time = time.time() - start

        # Calculate recall@10
        temp_results = {}
        for i, qid in enumerate(query_id_list):
            retrieved_indices = labels[i]
            retrieved_doc_ids = [corpus_ids[idx] for idx in retrieved_indices]
            temp_results[qid] = retrieved_doc_ids

        # Convert to list format for metrics
        retrieved_ids = []
        relevant_ids = []
        for qid in temp_results.keys():
            retrieved_ids.append(temp_results[qid])
            relevant_ids.append(list(qrels.get(qid, {}).keys()))

        recall_dict = compute_recall_at_k(retrieved_ids, relevant_ids, k_values=[10])
        recall = recall_dict[10]

        ef_results.append(
            {
                "ef": ef,
                "recall@10": recall,
                "search_time": search_time,
                "qps": len(query_emb) / search_time,
            }
        )

        print(
            f"efSearch={ef:3d}: Recall@10={recall:.4f}, Time={search_time:.3f}s, QPS={len(query_emb)/search_time:.1f}"
        )

    return ef_results


ef_results = analyze_ef_parameter(index, query_emb, query_id_list, corpus_ids, qrels)


# %% Plot Results
def plot_ef_tradeoff(ef_results, output_dir):
    """Plot efSearch vs recall and latency."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    efs = [r["ef"] for r in ef_results]
    recalls = [r["recall@10"] for r in ef_results]
    times = [r["search_time"] * 1000 / N_QUERY_SAMPLES for r in ef_results]

    ax1.plot(efs, recalls, marker="o", linewidth=2, markersize=8)
    ax1.set_xlabel("efSearch", fontsize=12)
    ax1.set_ylabel("Recall@10", fontsize=12)
    ax1.set_title("efSearch vs Recall@10", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(efs, times, marker="o", color="orange", linewidth=2, markersize=8)
    ax2.set_xlabel("efSearch", fontsize=12)
    ax2.set_ylabel("Latency (ms per query)", fontsize=12)
    ax2.set_title("efSearch vs Search Latency", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = f"{output_dir}/ef_tradeoff.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nPlot saved to: {output_path}")


plot_ef_tradeoff(ef_results, FAISS_REPORTS)


# %% [markdown]
# ## Save Metrics Report


# %% Save Report
def save_report(
    build_time,
    search_time,
    recall_scores,
    precision_scores,
    mrr_score,
    ef_results,
    output_dir,
):
    """Save metrics report to JSON."""
    dim = corpus_embeddings.shape[1]

    metrics_report = {
        "method": "faiss_hnsw",
        "search_latency": {
            "total_seconds": float(search_time),
            "queries_per_second": float(len(query_id_list) / search_time),
            "avg_latency_ms": float(search_time * 1000 / len(query_id_list)),
        },
        "build_latency": {
            "total_seconds": float(build_time),
            "vectors_per_second": float(len(corpus_ids) / build_time) if build_time > 0 else None,
        },
        "recall": {str(k): float(v) for k, v in recall_scores.items()},
        "precision": {str(k): float(v) for k, v in precision_scores.items()},
        "mrr": float(mrr_score),
        "metadata": {
            "dataset": DATASET_NAME,
            "use_subset": USE_SUBSET,
            "subset_size": SUBSET_SIZE if USE_SUBSET else None,
            "split": SPLIT,
            "n_corpus_docs": len(corpus_ids),
            "n_queries": len(query_id_list),
            "embedding_dimension": dim,
            "faiss_params": {
                "M": M,
                "efConstruction": EF_CONSTRUCTION,
                "efSearch": EF_SEARCH,
            },
        },
        "ef_sensitivity": ef_results,
    }

    report_path = save_metrics_report("faiss_hnsw", metrics_report, output_dir)
    print(f"\nMetrics report saved to: {report_path}")


save_report(
    build_time,
    search_time,
    recall_scores,
    precision_scores,
    mrr_score,
    ef_results,
    FAISS_REPORTS,
)


# %% [markdown]
# %% Summary
print(f"\n{'=' * 80}")
print(f"FAISS HNSW BENCHMARK COMPLETE")
print(f"{'=' * 80}\n")
if build_time > 0:
    print(f"Index: {len(corpus_ids):,} vectors built in {build_time:.1f}s")
else:
    print(f"Index: {len(corpus_ids):,} vectors (loaded from disk)")
print(f"Search: {len(query_id_list):,} queries in {search_time:.2f}s")
print(f"Recall@10: {recall_scores[10]:.4f}")
print(f"Latency: {search_time * 1000 / len(query_id_list):.2f} ms/query")
print(f"\nResults saved to {FAISS_REPORTS}/")
print(f"  - ef_tradeoff.png (parameter tuning analysis)")
print(f"  - faiss_hnsw_metrics.json (full performance report)")

# %% [markdown]
# expected outputs (might vary with different machine and parameters)

# ```
# ================================================================================
# BUILDING FAISS HNSW INDEX
# ================================================================================

# Dataset: msmarco (1M subset)
# Corpus size: 1,000,000 documents
# Dimension: 384

# FAISS HNSW Parameters:
#   M (connections per layer): 32
#   efConstruction: 100
#   Metric: Inner Product (for normalized vectors = cosine)

# Normalizing embeddings for cosine similarity...

# Building index (batch size: 1,000)...
#   Progress: 100.0% (1,000,000/1,000,000) - 380.0s elapsed

# Index built in 380.01 seconds (6.3 minutes)
# Build speed: 2632 vectors/sec

# Index Statistics:
#   Total elements: 1,000,000

# Index saved to: ../data/index/faiss_index.bin
# ```

# ```
# ================================================================================
# ANALYZING FAISS HNSW INDEX BEHAVIOR
# ================================================================================

# Demonstrating FAISS HNSW search with different efSearch values:
# (Shows how efSearch trades off speed vs accuracy)

# efSearch= 10: 2.28 ms/query
#   Query 0 top-3 neighbors: [927045 489818 193960] (distances: [0.441 0.561 0.583])
# efSearch= 50: 20.16 ms/query
#   Query 0 top-3 neighbors: [927045 489818 193960] (distances: [0.441 0.561 0.583])
# efSearch=200: 19.95 ms/query
#   Query 0 top-3 neighbors: [927045 489818 193960] (distances: [0.441 0.561 0.583])

# ** FAISS HNSW Properties **
# - Lower efSearch → Faster search (explores fewer nodes)
# - Higher efSearch → More accurate (explores more of the graph)
# - FAISS uses inner product metric (equivalent to cosine after normalization)
# ```

# ```

# ================================================================================
# FAISS HNSW SEARCH
# ================================================================================

# Search parameters:
#   efSearch: 50
#   K (neighbors to retrieve): 100

# Searching 100 queries...
# Search completed in 0.01 seconds
# Queries per second: 11805.0
# Avg latency per query: 0.08 ms

#```

# ```
# ================================================================================
# FAISS HNSW BENCHMARK COMPLETE
# ================================================================================

# Index: 1,000,000 vectors built in 380.0s
# Search: 100 queries in 0.01s
# Recall@10: 0.7683
# Latency: 0.08 ms/query

# Results saved to ../reports/faiss/
#   - ef_tradeoff.png (parameter tuning analysis)
#   - faiss_hnsw_metrics.json (full performance report)
# ```