"""
Shared utilities for vector search benchmarking.

This module provides standardized metrics computation and logging
for all vector search methods (brute-force, HNSW, FAISS, Qdrant).
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


def compute_recall_at_k(
    retrieved_ids: List[List[str]],
    relevant_ids: List[List[str]],
    k_values: List[int] = [1, 5, 10, 20, 50, 100],
) -> Dict[int, float]:
    """
    Compute Recall@K for retrieved results.

    Recall@K = (# relevant docs in top-K) / (total # relevant docs)

    Args:
        retrieved_ids: List of retrieved doc IDs per query [[doc1, doc2, ...], ...]
        relevant_ids: List of ground truth relevant doc IDs per query [[doc1, ...], ...]
        k_values: List of K values to compute recall for

    Returns:
        Dictionary mapping K -> Recall@K score
    """
    recalls = {k: [] for k in k_values}

    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        relevant_set = set(relevant)
        if len(relevant_set) == 0:
            continue

        for k in k_values:
            retrieved_at_k = set(retrieved[:k])
            recall = len(retrieved_at_k & relevant_set) / len(relevant_set)
            recalls[k].append(recall)

    # Average across all queries
    return {k: np.mean(scores) if scores else 0.0 for k, scores in recalls.items()}


def compute_precision_at_k(
    retrieved_ids: List[List[str]],
    relevant_ids: List[List[str]],
    k_values: List[int] = [1, 5, 10],
) -> Dict[int, float]:
    """
    Compute Precision@K for retrieved results.

    Precision@K = (# relevant docs in top-K) / K

    Args:
        retrieved_ids: List of retrieved doc IDs per query
        relevant_ids: List of ground truth relevant doc IDs per query
        k_values: List of K values to compute precision for

    Returns:
        Dictionary mapping K -> Precision@K score
    """
    precisions = {k: [] for k in k_values}

    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        relevant_set = set(relevant)

        for k in k_values:
            retrieved_at_k = retrieved[:k]
            if len(retrieved_at_k) == 0:
                precisions[k].append(0.0)
            else:
                precision = len(set(retrieved_at_k) & relevant_set) / len(
                    retrieved_at_k
                )
                precisions[k].append(precision)

    return {k: np.mean(scores) if scores else 0.0 for k, scores in precisions.items()}


def compute_mrr(retrieved_ids: List[List[str]], relevant_ids: List[List[str]]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    MRR = Average of (1 / rank of first relevant doc)

    Args:
        retrieved_ids: List of retrieved doc IDs per query
        relevant_ids: List of ground truth relevant doc IDs per query

    Returns:
        MRR score
    """
    reciprocal_ranks = []

    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        relevant_set = set(relevant)

        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_set:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def save_metrics_report(
    method_name: str, metrics: Dict[str, Any], reports_dir: str = "../reports"
):
    """
    Save metrics to JSON report file.

    Args:
        method_name: Name of search method (e.g., 'brute_force', 'hnsw', 'faiss')
        metrics: Dictionary of metrics and metadata
        reports_dir: Directory to save reports
    """
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{method_name}_{timestamp}.json"
    filepath = reports_path / filename

    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {filepath}")
    return str(filepath)


def format_metrics_table(metrics: Dict[str, Any]) -> str:
    """
    Format metrics as a readable table string.

    Args:
        metrics: Dictionary of metrics

    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 60)
    lines.append("METRICS SUMMARY")
    lines.append("=" * 60)

    # Performance metrics
    if "performance" in metrics:
        lines.append("\nPerformance:")
        for key, value in metrics["performance"].items():
            if "_ms" in key.lower():
                lines.append(f"  {key}: {value:.4f} ms")
            elif "time" in key.lower():
                lines.append(f"  {key}: {value:.4f} seconds")
            elif "qps" in key.lower():
                lines.append(f"  {key}: {value:.2f}")
            else:
                lines.append(f"  {key}: {value}")

    # Recall metrics
    if "recall" in metrics:
        lines.append("\nRecall@K:")
        for k, score in sorted(metrics["recall"].items()):
            lines.append(f"  Recall@{k}: {score:.4f}")

    # Precision metrics
    if "precision" in metrics:
        lines.append("\nPrecision@K:")
        for k, score in sorted(metrics["precision"].items()):
            lines.append(f"  Precision@{k}: {score:.4f}")

    # MRR
    if "mrr" in metrics:
        lines.append(f"\nMRR: {metrics['mrr']:.4f}")

    # Metadata
    if "metadata" in metrics:
        lines.append("\nMetadata:")
        for key, value in metrics["metadata"].items():
            lines.append(f"  {key}: {value}")

    lines.append("=" * 60)

    return "\n".join(lines)


class BenchmarkTimer:
    """Context manager for timing operations."""

    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        print(f"{self.description}: {self.elapsed:.4f} seconds")
