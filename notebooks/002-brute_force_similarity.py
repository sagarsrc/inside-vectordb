# %% [markdown]
# # Brute-Force Vector Similarity Search
#
# This notebook implements baseline brute-force cosine similarity search.
# It serves as the ground truth for comparing more advanced methods (HNSW, FAISS, Qdrant).
#
# **Method**: Compute cosine similarity between query and ALL corpus documents, then sort.
#
# **Metrics**: Recall@K, Precision@K, MRR, Search latency
#
# **Note**: Using subset of data for faster experimentation

# %% [markdown]
# ## Configuration

# %% Global Configuration
DATA_ROOT = "../data"
REPORTS_DIR = "../reports/brute_force"
DATASET_NAME = "msmarco"
USE_SUBSET = True  # Use 1M subset for faster experimentation
SUBSET_SIZE = "1M"  # Which subset to use
SPLIT = "dev"  # For MS MARCO: 'dev' (6,980 queries) or 'test' (43 queries)

# Sampling for faster testing
N_CORPUS_SAMPLES = None  # Use entire corpus (None = all documents)
N_QUERY_SAMPLES = 100  # Use subset of queries for testing (set to None for all 6,980)

# Search parameters
K_VALUES = [1, 5, 10, 20, 50, 100]  # K values for Recall@K and Precision@K

# %% [markdown]
# ## Import Dependencies

# %% Imports
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import time
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from beir.datasets.data_loader import GenericDataLoader

# Import shared utilities
from utils import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr,
    save_metrics_report,
    format_metrics_table,
    BenchmarkTimer,
)

# %% [markdown]
# ## Load Embeddings and Ground Truth


# %% Load Embeddings
def load_embeddings(npz_path: str):
    """Load embeddings from NPZ file."""
    print(f"Loading: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    ids = data["ids"].tolist()  # Convert to list
    print(f"  Loaded {len(ids):,} embeddings with dimension {embeddings.shape[1]}")
    return embeddings, ids


# Load corpus and query embeddings
subset_suffix = f"_{SUBSET_SIZE}" if USE_SUBSET else ""
corpus_emb_path = f"{DATA_ROOT}/{DATASET_NAME}{subset_suffix}_corpus_embeddings.npz"
query_emb_path = f"{DATA_ROOT}/{DATASET_NAME}{subset_suffix}_query_embeddings.npz"

corpus_embeddings, corpus_ids = load_embeddings(corpus_emb_path)
query_embeddings, query_ids = load_embeddings(query_emb_path)


# %% Load Ground Truth (Qrels)
def load_qrels(data_root: str, dataset_name: str, split: str = "dev"):
    """Load ground truth relevance judgments."""
    import json

    dataset_path = f"{data_root}/{dataset_name}"

    # Load queries directly from file
    queries_file = f"{dataset_path}/queries.jsonl"
    queries = {}
    with open(queries_file, "r") as f:
        for line in f:
            query = json.loads(line)
            queries[query["_id"]] = query["text"]

    # Load qrels directly from file
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

    return queries, qrels


queries_dict, qrels = load_qrels(DATA_ROOT, DATASET_NAME, split=SPLIT)

# %% [markdown]
# ## Sample Data for Testing


# %% Sample Corpus and Queries
def sample_data(
    corpus_embeddings: np.ndarray,
    corpus_ids: list,
    query_embeddings: np.ndarray,
    query_ids: list,
    qrels: dict,
    n_corpus: int,
    n_queries: int,
):
    """Sample subset of data for faster testing."""

    # Use entire corpus (no sampling)
    if n_corpus is None:
        corpus_sample_emb = corpus_embeddings
        corpus_sample_ids = corpus_ids
    else:
        n_corpus = min(n_corpus, len(corpus_ids))
        corpus_sample_emb = corpus_embeddings[:n_corpus]
        corpus_sample_ids = corpus_ids[:n_corpus]

    # Sample queries that have qrels
    query_ids_with_qrels = [qid for qid in query_ids if qid in qrels]
    n_queries = min(n_queries, len(query_ids_with_qrels))
    sampled_query_ids = query_ids_with_qrels[:n_queries]

    # Get embeddings for sampled queries
    query_id_to_idx = {qid: idx for idx, qid in enumerate(query_ids)}
    sampled_indices = [query_id_to_idx[qid] for qid in sampled_query_ids]
    query_sample_emb = query_embeddings[sampled_indices]

    print(f"\nSampled data:")
    print(f"  Corpus: {len(corpus_sample_ids):,} documents")
    print(f"  Queries: {len(sampled_query_ids):,} queries")

    return (corpus_sample_emb, corpus_sample_ids, query_sample_emb, sampled_query_ids)


corpus_emb, corpus_id_list, query_emb, query_id_list = sample_data(
    corpus_embeddings,
    corpus_ids,
    query_embeddings,
    query_ids,
    qrels,
    N_CORPUS_SAMPLES,
    N_QUERY_SAMPLES,
)

# %% [markdown]
# ## Implement Brute-Force Search


# %% Brute-Force Cosine Similarity Search
def brute_force_search(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus_ids: list,
    top_k: int = 100,
) -> list:
    """
    Brute-force search using cosine similarity.

    Args:
        query_embedding: Query vector (1D array)
        corpus_embeddings: Corpus vectors (2D array)
        corpus_ids: List of document IDs
        top_k: Number of top results to return

    Returns:
        List of top-k document IDs
    """
    # Compute cosine similarity with all documents
    similarities = cosine_similarity(query_embedding.reshape(1, -1), corpus_embeddings)[
        0
    ]

    # Get top-k indices
    top_k_indices = np.argsort(similarities)[::-1][:top_k]

    # Return corresponding doc IDs
    return [corpus_ids[idx] for idx in top_k_indices]


def batch_brute_force_search(
    query_embeddings: np.ndarray,
    query_ids: list,
    corpus_embeddings: np.ndarray,
    corpus_ids: list,
    top_k: int = 100,
) -> dict:
    """
    Run brute-force search for multiple queries.

    Args:
        query_embeddings: Query vectors (2D array)
        query_ids: List of query IDs
        corpus_embeddings: Corpus vectors (2D array)
        corpus_ids: List of document IDs
        top_k: Number of results per query

    Returns:
        Dictionary mapping query_id -> list of retrieved doc_ids
    """
    results = {}

    for i, query_id in enumerate(query_ids):
        retrieved_ids = brute_force_search(
            query_embeddings[i], corpus_embeddings, corpus_ids, top_k=top_k
        )
        results[query_id] = retrieved_ids

    return results


# %% [markdown]
# ## Run Search and Measure Performance

# %% Execute Brute-Force Search
print("\n" + "=" * 60)
print("RUNNING BRUTE-FORCE SEARCH")
print("=" * 60)

with BenchmarkTimer("Total search time") as timer:
    search_results = batch_brute_force_search(
        query_emb, query_id_list, corpus_emb, corpus_id_list, top_k=max(K_VALUES)
    )

total_time = timer.elapsed
avg_query_time = total_time / len(query_id_list)
qps = len(query_id_list) / total_time

print(f"Average query time: {avg_query_time * 1000:.2f} ms")
print(f"Queries per second: {qps:.2f}")

# %% [markdown]
# ## Display Sample Results


# %% Show Sample Search Results
def display_sample_results(
    search_results: dict,
    queries_dict: dict,
    qrels: dict,
    n_samples: int = 3,
    top_k: int = 5,
):
    """Display sample search results with ground truth comparison."""
    print("\n" + "=" * 80)
    print("SAMPLE SEARCH RESULTS")
    print("=" * 80)

    sample_query_ids = list(search_results.keys())[:n_samples]

    for query_id in sample_query_ids:
        print(f"\nQuery ID: {query_id}")
        print(f"Query: {queries_dict[query_id]}")

        # Ground truth
        gt_docs = list(qrels.get(query_id, {}).keys())
        print(f"\nGround Truth ({len(gt_docs)} docs): {gt_docs}")

        # Retrieved
        retrieved = search_results[query_id][:top_k]
        print(f"\nTop-{top_k} Retrieved:")
        for i, doc_id in enumerate(retrieved, 1):
            is_relevant = "" if doc_id in gt_docs else ""
            print(f"  {i}. {doc_id} {is_relevant}")

        print("-" * 80)


display_sample_results(search_results, queries_dict, qrels, n_samples=3, top_k=10)

# %% [markdown]
# ## Compute Evaluation Metrics


# %% Prepare Data for Metrics
def prepare_metrics_data(search_results: dict, qrels: dict):
    """
    Convert search results and qrels to lists for metrics computation.

    Args:
        search_results: Dict of query_id -> retrieved doc_ids
        qrels: Dict of query_id -> {doc_id: relevance_score}

    Returns:
        Tuple of (retrieved_ids_list, relevant_ids_list)
    """
    retrieved_ids_list = []
    relevant_ids_list = []

    for query_id in search_results.keys():
        retrieved_ids_list.append(search_results[query_id])

        # Get relevant doc IDs from qrels
        relevant_docs = list(qrels.get(query_id, {}).keys())
        relevant_ids_list.append(relevant_docs)

    return retrieved_ids_list, relevant_ids_list


retrieved_ids, relevant_ids = prepare_metrics_data(search_results, qrels)

# %% Compute Metrics
print("\n" + "=" * 60)
print("COMPUTING METRICS")
print("=" * 60)

# Recall@K
recall_scores = compute_recall_at_k(retrieved_ids, relevant_ids, k_values=K_VALUES)
print("\nRecall@K:")
for k, score in sorted(recall_scores.items()):
    print(f"  Recall@{k}: {score:.4f}")

# Precision@K
precision_scores = compute_precision_at_k(
    retrieved_ids, relevant_ids, k_values=[1, 5, 10]
)
print("\nPrecision@K:")
for k, score in sorted(precision_scores.items()):
    print(f"  Precision@{k}: {score:.4f}")

# MRR
mrr_score = compute_mrr(retrieved_ids, relevant_ids)
print(f"\nMRR: {mrr_score:.4f}")

# %% [markdown]
# ## Save Metrics Report

# %% Create and Save Metrics Report
metrics_report = {
    "method": "brute_force",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "performance": {
        "total_search_time_seconds": total_time,
        "avg_query_time_seconds": avg_query_time,
        "avg_query_time_ms": avg_query_time * 1000,
        "queries_per_second": qps,
    },
    "recall": {str(k): float(v) for k, v in recall_scores.items()},
    "precision": {str(k): float(v) for k, v in precision_scores.items()},
    "mrr": float(mrr_score),
    "metadata": {
        "dataset": DATASET_NAME,
        "use_subset": USE_SUBSET,
        "subset_size": SUBSET_SIZE if USE_SUBSET else None,
        "split": SPLIT,
        "n_corpus_docs": len(corpus_id_list),
        "n_queries": len(query_id_list),
        "embedding_dimension": corpus_emb.shape[1],
        "top_k_retrieved": max(K_VALUES),
    },
}

# Save to JSON
report_path = save_metrics_report("brute_force", metrics_report, REPORTS_DIR)

# Display formatted summary
print("\n" + format_metrics_table(metrics_report))

# %% [markdown]
# ## Summary
#
# **Brute-Force Search Characteristics**:
# - **Pros**: 100% accurate (exact search), simple implementation
# - **Cons**: Slow for large datasets (O(n) per query), doesn't scale
#
# **Next Steps**:
# - Compare with HNSW (approximate but much faster)
# - Compare with FAISS (optimized exact + approximate)
# - Compare with Qdrant (production-ready vector DB)

# %%
