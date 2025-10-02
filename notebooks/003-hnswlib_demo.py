# %% [markdown]
# # HNSWlib Demo: Fast Approximate Nearest Neighbor Search
#
# This notebook demonstrates **Hierarchical Navigable Small Worlds (HNSW)** using hnswlib.
#
# **What You'll Learn:**
# - How HNSW builds a multi-layer graph structure
# - Key parameters: M (connections), ef_construction, ef_search
# - Performance comparison vs brute-force
#
# **Why HNSWlib?**
# - Pure HNSW implementation (no abstractions)
# - Fast C++ backend with Python bindings
# - Full control over parameters
# - Used in production (Qdrant, Weaviate use similar algorithms)

# %% [markdown]
# ## Configuration

# %% Global Configuration
import os

DATA_ROOT = "../data"
REPORTS_DIR = "../reports"
INDEX_DIR = "../data/index"
INDEX_PATH = "../data/index/hnswlib_index.bin"
DATASET_NAME = "msmarco"
USE_SUBSET = True
SUBSET_SIZE = "1M"
SPLIT = "dev"

# HNSW Parameters
M = 16  # Max connections per node (higher = better recall, more memory)
EF_CONSTRUCTION = 100  # Build-time accuracy (higher = better index quality, slower build)
EF_SEARCH = 50  # Search-time accuracy (higher = better recall, slower search)

# Benchmark settings
N_QUERY_SAMPLES = 100  # Number of queries to test
K_VALUES = [1, 5, 10, 20, 50, 100]  # K for recall@K evaluation

# %% [markdown]
# ## Import Dependencies

# %% Imports
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import hnswlib
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Import shared utilities
from utils import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr,
    save_metrics_report,
)

# Check hnswlib is available
print("hnswlib imported successfully")

# Create reports directory structure
HNSWLIB_REPORTS = f"{REPORTS_DIR}/hnswlib"
Path(HNSWLIB_REPORTS).mkdir(parents=True, exist_ok=True)
print(f"Reports will be saved to: {HNSWLIB_REPORTS}")


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
    # Filter to only queries that have ground truth
    query_indices = []
    query_id_list = []

    for i, qid in enumerate(query_ids):
        if qid in qrels:
            query_indices.append(i)
            query_id_list.append(qid)

    # Sample subset if requested
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
# ## Build HNSW Index
#
# **Key Parameters:**
# - `M = 16`: Each node connects to max 16 neighbors (trade-off: recall vs memory)
# - `ef_construction = 100`: Build-time search width (higher = better quality, slower)
#   - Reduced from 200 to 100 for **2x faster** build time with minimal quality loss
#   - For 1M docs: ~3-5 minutes instead of 10+ minutes
# - `space = 'cosine'`: Distance metric (cosine similarity for text embeddings)
# - `num_threads = -1`: Use all CPU cores for parallel processing
#
# **Build Optimizations:**
# - Batch insertion (50K vectors/batch) for better memory locality
# - Progress tracking to monitor build status
# - Parallel processing across all CPU cores


# %% HNSW Index Building
def build_hnsw_index(embeddings, m, ef_construction, num_threads=-1):
    """Build HNSW index with parallel processing.

    Args:
        embeddings: Embedding vectors
        m: Max connections per node
        ef_construction: Build-time search width
        num_threads: Number of CPU threads (-1 = use all available cores)
    """
    dim = embeddings.shape[1]
    n_elements = len(embeddings)

    print(f"\n{'=' * 80}")
    print(f"BUILDING HNSW INDEX")
    print(f"{'=' * 80}\n")

    print(f"Dataset: {DATASET_NAME} ({SUBSET_SIZE} subset)" if USE_SUBSET else DATASET_NAME)
    print(f"Corpus size: {n_elements:,} documents")
    print(f"Dimension: {dim}")
    print(f"\nHNSW Parameters:")
    print(f"  M (max connections): {m}")
    print(f"  ef_construction: {ef_construction}")
    print(f"  num_threads: {num_threads} (all cores)" if num_threads == -1 else f"  num_threads: {num_threads}")
    print(f"  Space: cosine")

    # Create index
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n_elements, ef_construction=ef_construction, M=m)

    # Set number of threads for parallel operations
    index.set_num_threads(num_threads)

    # Add items in batches with progress tracking
    batch_size = 1000  # Batch size for progress updates
    start_time = time.time()

    print(f"\nBuilding index (batch size: {batch_size:,})...")
    for i in range(0, n_elements, batch_size):
        end_idx = min(i + batch_size, n_elements)
        batch_embeddings = embeddings[i:end_idx]
        batch_ids = np.arange(i, end_idx)
        index.add_items(batch_embeddings, batch_ids)

        # Progress update
        progress = (end_idx / n_elements) * 100
        elapsed = time.time() - start_time
        print(f"  Progress: {progress:5.1f}% ({end_idx:,}/{n_elements:,}) - {elapsed:.1f}s elapsed", end="\r")

    build_time = time.time() - start_time
    print()  # New line after progress
    print(f"\nIndex built in {build_time:.2f} seconds ({build_time/60:.1f} minutes)")
    print(f"Build speed: {n_elements / build_time:.0f} vectors/sec")
    print(f"\nIndex Statistics:")
    print(f"  Total elements: {index.get_current_count():,}")
    print(f"  Max elements: {index.get_max_elements():,}")

    return index, build_time


# Load or build HNSW index
index_file = Path(INDEX_PATH)
if index_file.exists():
    print(f"\n{'=' * 80}")
    print(f"LOADING EXISTING HNSW INDEX")
    print(f"{'=' * 80}\n")
    print(f"Loading index from: {index_file}")

    dim = corpus_embeddings.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(str(index_file), max_elements=len(corpus_embeddings))
    index.set_num_threads(-1)

    print(f"Index loaded successfully")
    print(f"  Total elements: {index.get_current_count():,}")
    print(f"  Max elements: {index.get_max_elements():,}")

    build_time = 0  # No build time since we loaded from disk
else:
    index, build_time = build_hnsw_index(corpus_embeddings, M, EF_CONSTRUCTION, num_threads=-1)

    # Save index to disk
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    index.save_index(str(index_file))
    print(f"\nIndex saved to: {index_file}")


# %% [markdown]
# ## Analyze Search Behavior
#
# Demonstrate how different ef values affect search quality and speed


# %% Search Behavior Analysis
def analyze_search_behavior(index, query_emb, query_id_list, corpus_ids):
    """Analyze HNSW search behavior with different ef values."""
    print(f"\n{'=' * 80}")
    print(f"ANALYZING ACTUAL HNSW INDEX BEHAVIOR")
    print(f"{'=' * 80}\n")

    sample_queries = query_emb[:5]
    ef_test_values = [10, 50, 200]
    k_test = 10

    print("Demonstrating HNSW search with different ef values:")
    print("(Shows how HNSW trades off speed vs accuracy)\n")

    for ef_val in ef_test_values:
        index.set_ef(ef_val)

        # Time search
        start = time.time()
        labels, distances = index.knn_query(sample_queries, k=k_test)
        search_time = (time.time() - start) * 1000 / len(sample_queries)

        print(f"ef={ef_val:3d}: {search_time:.2f} ms/query")
        print(
            f"  Query 0 top-3 neighbors: {labels[0][:3]} (distances: {distances[0][:3].round(3)})"
        )

    print("\n** HNSW Properties Demonstrated **")
    print("- Lower ef → Faster search (explores fewer nodes)")
    print("- Higher ef → More accurate (explores more of the graph)")
    print("- Graph structure is hierarchical (not exposed in Python API)")
    print("- Real vector databases (Qdrant/Weaviate) expose similar parameters")


analyze_search_behavior(index, query_emb, query_id_list, corpus_ids)


# %% [markdown]
# ## Search with HNSW
#
# **Search Parameter:**
# - `ef = 50`: Search-time exploration width
#   - Higher ef = better recall, slower search
#   - Lower ef = faster search, lower recall


# %% HNSW Search
def perform_hnsw_search(index, query_emb, query_id_list, corpus_ids, ef_search, k_max):
    """Perform HNSW search on all queries."""
    print(f"\n{'=' * 80}")
    print(f"HNSW SEARCH")
    print(f"{'=' * 80}\n")

    print(f"Search parameters:")
    print(f"  ef (search width): {ef_search}")
    print(f"  K (neighbors to retrieve): {k_max}")

    # Set search parameter
    index.set_ef(ef_search)

    # Perform search with timing
    print(f"\nSearching {len(query_emb):,} queries...")
    start_time = time.time()
    labels, distances = index.knn_query(query_emb, k=k_max)
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


all_results, search_time = perform_hnsw_search(
    index, query_emb, query_id_list, corpus_ids, EF_SEARCH, max(K_VALUES)
)


# %% [markdown]
# ## Evaluate Performance
#
# **Metrics:**
# - **Recall@K**: % of relevant docs found in top K
# - **Precision@K**: % of top K that are relevant
# - **MRR**: Mean Reciprocal Rank of first relevant doc


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
    print(f"HNSW PERFORMANCE METRICS")
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
#
# Demonstrate how `ef` affects search quality and speed


# %% Parameter Analysis
def analyze_ef_parameter(index, query_emb, query_id_list, corpus_ids, qrels):
    """Analyze ef parameter sensitivity."""
    print(f"\n{'=' * 80}")
    print(f"PARAMETER SENSITIVITY: ef (search width)")
    print(f"{'=' * 80}\n")

    ef_values = [10, 20, 50, 100, 200, 500]
    ef_results = []

    for ef in ef_values:
        index.set_ef(ef)

        # Time search
        start = time.time()
        labels, _ = index.knn_query(query_emb, k=10)
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
            f"ef={ef:3d}: Recall@10={recall:.4f}, Time={search_time:.3f}s, QPS={len(query_emb)/search_time:.1f}"
        )

    return ef_results


ef_results = analyze_ef_parameter(index, query_emb, query_id_list, corpus_ids, qrels)


# %% Plot Results
def plot_ef_tradeoff(ef_results, output_dir):
    """Plot ef vs recall and latency."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    efs = [r["ef"] for r in ef_results]
    recalls = [r["recall@10"] for r in ef_results]
    times = [
        r["search_time"] * 1000 / N_QUERY_SAMPLES for r in ef_results
    ]  # ms per query

    ax1.plot(efs, recalls, marker="o", linewidth=2, markersize=8)
    ax1.set_xlabel("ef (search width)", fontsize=12)
    ax1.set_ylabel("Recall@10", fontsize=12)
    ax1.set_title("ef vs Recall@10", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(efs, times, marker="o", color="orange", linewidth=2, markersize=8)
    ax2.set_xlabel("ef (search width)", fontsize=12)
    ax2.set_ylabel("Latency (ms per query)", fontsize=12)
    ax2.set_title("ef vs Search Latency", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = f"{output_dir}/ef_tradeoff.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nPlot saved to: {output_path}")


plot_ef_tradeoff(ef_results, HNSWLIB_REPORTS)


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
        "method": "hnswlib",
        "search_latency": {
            "total_seconds": float(search_time),
            "queries_per_second": float(len(query_id_list) / search_time),
            "avg_latency_ms": float(search_time * 1000 / len(query_id_list)),
        },
        "build_latency": {
            "total_seconds": float(build_time),
            "vectors_per_second": float(len(corpus_ids) / build_time),
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
            "hnsw_params": {
                "M": M,
                "ef_construction": EF_CONSTRUCTION,
                "ef_search": EF_SEARCH,
            },
        },
        "ef_sensitivity": ef_results,
    }

    report_path = save_metrics_report("hnswlib", metrics_report, output_dir)
    print(f"\nMetrics report saved to: {report_path}")


save_report(
    build_time,
    search_time,
    recall_scores,
    precision_scores,
    mrr_score,
    ef_results,
    HNSWLIB_REPORTS,
)



# %% Summary
print(f"\n{'=' * 80}")
print(f"HNSW DEMO COMPLETE")
print(f"{'=' * 80}\n")
print(f"Index: {len(corpus_ids):,} vectors in {build_time:.1f}s")
print(f"Search: {len(query_id_list):,} queries in {search_time:.2f}s")
print(f"Recall@10: {recall_scores[10]:.4f}")
print(f"Latency: {search_time * 1000 / len(query_id_list):.2f} ms/query")
print(f"\nResults saved to {HNSWLIB_REPORTS}/")
print(f"  - ef_tradeoff.png (parameter tuning analysis)")
print(f"  - hnswlib_metrics.json (full performance report)")



# %% [markdown]
# expected outputs (might vary with different machine and parameters)

# ```
# ================================================================================
# BUILDING HNSW INDEX
# ================================================================================

# Dataset: msmarco (1M subset)
# Corpus size: 1,000,000 documents
# Dimension: 384

# HNSW Parameters:
#   M (max connections): 16
#   ef_construction: 100
#   num_threads: -1 (all cores)
#   Space: cosine

# Building index (batch size: 1,000)...
#   Progress: 100.0% (1,000,000/1,000,000) - 375.7s elapsed

# Index built in 375.71 seconds (6.3 minutes)
# Build speed: 2662 vectors/sec

# Index Statistics:
#   Total elements: 1,000,000
#   Max elements: 1,000,000
# ```
#
# ```

# ================================================================================
# ANALYZING ACTUAL HNSW INDEX BEHAVIOR
# ================================================================================

# Demonstrating HNSW search with different ef values:
# (Shows how HNSW trades off speed vs accuracy)

# ef= 10: 0.09 ms/query
#   Query 0 top-3 neighbors: [927045 489818 193960] (distances: [0.221 0.28  0.291])
# ef= 50: 0.16 ms/query
#   Query 0 top-3 neighbors: [927045 489818 193960] (distances: [0.221 0.28  0.291])
# ef=200: 0.52 ms/query
#   Query 0 top-3 neighbors: [927045 489818 193960] (distances: [0.221 0.28  0.291])

# ** HNSW Properties Demonstrated **
# - Lower ef → Faster search (explores fewer nodes)
# - Higher ef → More accurate (explores more of the graph)
# - Graph structure is hierarchical (not exposed in Python API)
# - Real vector databases (Qdrant/Weaviate) expose similar parameters
# ```

# ```
# ================================================================================
# HNSW SEARCH
# ================================================================================

# Search parameters:
#   ef (search width): 50
#   K (neighbors to retrieve): 100

# Searching 100 queries...
# Search completed in 0.04 seconds
# Queries per second: 2274.8
# Avg latency per query: 0.44 ms
# ```

# ```
# ================================================================================
# HNSW DEMO COMPLETE
# ================================================================================

# Index: 1,000,000 vectors in 375.7s
# Search: 100 queries in 0.04s
# Recall@10: 0.7583
# Latency: 0.44 ms/query

# Results saved to ../reports/hnswlib/
#   - ef_tradeoff.png (parameter tuning analysis)
#   - hnswlib_metrics.json (full performance report)
# ```
