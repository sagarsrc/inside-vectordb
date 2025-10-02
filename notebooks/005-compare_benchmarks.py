# %% [markdown]
# # Benchmark Comparison: Brute Force vs HNSWlib vs FAISS
#
# This notebook compares all vector search methods implemented so far:
# - **Brute Force**: Exact search baseline (100% recall)
# - **HNSWlib**: Pure HNSW implementation
# - **FAISS**: Meta's production library with HNSW
#
# **What We'll Compare:**
# - Search latency (ms per query)
# - Build time (index construction)
# - Recall@K (search quality)
# - Speed vs Accuracy trade-offs

# %% [markdown]
# ## Configuration

# %% Global Configuration
import os
import json
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

REPORTS_DIR = "../reports"
OUTPUT_DIR = "../reports/summary"

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

print("Benchmark Comparison Tool")
print("=" * 80)
print(f"Loading reports from: {REPORTS_DIR}")
print(f"Saving outputs to: {OUTPUT_DIR}")

# %% [markdown]
# ## Load All Benchmark Reports

# %% Load Reports
def load_latest_reports(reports_dir):
    """Load the most recent report from each method directory."""
    reports = {}

    # Find all subdirectories in reports/
    for method_dir in Path(reports_dir).iterdir():
        if not method_dir.is_dir():
            continue

        # Find all JSON files in this method directory
        json_files = list(method_dir.glob("*.json"))

        if not json_files:
            continue

        # Get the most recent file (by modification time)
        latest_file = max(json_files, key=lambda f: f.stat().st_mtime)

        # Load the JSON
        with open(latest_file, 'r') as f:
            data = json.load(f)

        method_name = method_dir.name
        reports[method_name] = {
            'data': data,
            'file': str(latest_file),
            'timestamp': latest_file.stat().st_mtime
        }

        print(f" Loaded {method_name}: {latest_file.name}")

    return reports


reports = load_latest_reports(REPORTS_DIR)
print(f"\nFound {len(reports)} benchmark reports")

# %% [markdown]
# ## Extract Comparison Metrics

# %% Extract Metrics
def extract_metrics(reports):
    """Extract key metrics from all reports for comparison."""
    comparison = {
        'methods': [],
        'search_latency_ms': [],
        'queries_per_second': [],
        'build_time_seconds': [],
        'recall_at_10': [],
        'recall_at_100': [],
        'mrr': [],
        'params': []
    }

    for method_name, report_info in reports.items():
        data = report_info['data']

        comparison['methods'].append(method_name)

        # Search latency
        if 'performance' in data:  # Brute force format
            comparison['search_latency_ms'].append(data['performance']['avg_query_time_ms'])
            comparison['queries_per_second'].append(data['performance']['queries_per_second'])
            comparison['build_time_seconds'].append(0)  # No build time for brute force
        elif 'search_latency' in data:  # HNSW/FAISS format
            comparison['search_latency_ms'].append(data['search_latency']['avg_latency_ms'])
            comparison['queries_per_second'].append(data['search_latency']['queries_per_second'])
            comparison['build_time_seconds'].append(data['build_latency']['total_seconds'])
        elif 'search_latency_cpu' in data:  # FAISS CPU/GPU format
            comparison['search_latency_ms'].append(data['search_latency_cpu']['avg_latency_ms'])
            comparison['queries_per_second'].append(data['search_latency_cpu']['queries_per_second'])
            comparison['build_time_seconds'].append(data['build_latency']['total_seconds'])

        # Recall metrics
        recall_data = data.get('recall', {})
        comparison['recall_at_10'].append(float(recall_data.get('10', 0)))
        comparison['recall_at_100'].append(float(recall_data.get('100', 0)))
        comparison['mrr'].append(data.get('mrr', 0))

        # Parameters
        metadata = data.get('metadata', {})
        if 'hnsw_params' in metadata:
            params = metadata['hnsw_params']
            param_str = f"M={params['M']}, ef_c={params['ef_construction']}, ef_s={params['ef_search']}"
        elif 'faiss_params' in metadata:
            params = metadata['faiss_params']
            param_str = f"M={params['M']}, ef_c={params['efConstruction']}, ef_s={params['efSearch']}"
        else:
            param_str = "exact"
        comparison['params'].append(param_str)

    return comparison


metrics = extract_metrics(reports)

# Display summary table
print(f"\n{'=' * 80}")
print(f"BENCHMARK SUMMARY")
print(f"{'=' * 80}\n")

print(f"{'Method':<15} {'Latency (ms)':<15} {'QPS':<12} {'Recall@10':<12} {'Build Time':<15}")
print("-" * 80)
for i, method in enumerate(metrics['methods']):
    latency = metrics['search_latency_ms'][i]
    qps = metrics['queries_per_second'][i]
    recall = metrics['recall_at_10'][i]
    build = metrics['build_time_seconds'][i]

    build_str = f"{build:.1f}s" if build > 0 else "N/A"

    print(f"{method:<15} {latency:<15.2f} {qps:<12.1f} {recall:<12.4f} {build_str:<15}")

# %% [markdown]
# ## Visualization 1: Search Latency Comparison

# %% Plot Search Latency
def plot_search_latency(metrics, output_dir):
    """Bar chart comparing search latency."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = metrics['methods']
    latencies = metrics['search_latency_ms']

    # Create bar chart
    bars = ax.bar(methods, latencies, color=['#e74c3c', '#3498db', '#2ecc71'])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} ms',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Latency (ms per query)', fontsize=12)
    ax.set_title('Search Latency Comparison', fontsize=14, fontweight='bold')
    ax.set_yscale('log')  # Log scale to show differences
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = f"{output_dir}/latency_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f" Saved: {output_path}")


plot_search_latency(metrics, OUTPUT_DIR)

# %% [markdown]
# ## Visualization 2: Queries Per Second (Throughput)

# %% Plot QPS
def plot_qps(metrics, output_dir):
    """Bar chart comparing queries per second."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = metrics['methods']
    qps_values = metrics['queries_per_second']

    # Create bar chart
    bars = ax.bar(methods, qps_values, color=['#e74c3c', '#3498db', '#2ecc71'])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f} QPS',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Queries Per Second (QPS)', fontsize=12)
    ax.set_title('Search Throughput Comparison', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = f"{output_dir}/qps_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f" Saved: {output_path}")


plot_qps(metrics, OUTPUT_DIR)

# %% [markdown]
# ## Visualization 3: Recall@10 Comparison

# %% Plot Recall
def plot_recall(metrics, output_dir):
    """Bar chart comparing recall@10."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = metrics['methods']
    recalls = metrics['recall_at_10']

    # Create bar chart
    bars = ax.bar(methods, recalls, color=['#e74c3c', '#3498db', '#2ecc71'])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Recall@10', fontsize=12)
    ax.set_title('Search Quality Comparison (Recall@10)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.3, label='90% recall')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = f"{output_dir}/recall_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f" Saved: {output_path}")


plot_recall(metrics, OUTPUT_DIR)

# %% [markdown]
# ## Visualization 4: Speed vs Accuracy Trade-off

# %% Plot Speed vs Accuracy
def plot_speed_vs_accuracy(metrics, output_dir):
    """Scatter plot showing speed vs accuracy trade-off."""
    fig, ax = plt.subplots(figsize=(10, 7))

    methods = metrics['methods']
    latencies = metrics['search_latency_ms']
    recalls = metrics['recall_at_10']

    # Create scatter plot
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    markers = ['o', 's', '^']

    for i, method in enumerate(methods):
        ax.scatter(latencies[i], recalls[i],
                  s=300, color=colors[i], marker=markers[i],
                  label=method, alpha=0.7, edgecolors='black', linewidth=2)

        # Add method name annotation
        ax.annotate(method,
                   xy=(latencies[i], recalls[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold')

    ax.set_xlabel('Search Latency (ms per query)', fontsize=12)
    ax.set_ylabel('Recall@10', fontsize=12)
    ax.set_title('Speed vs Accuracy Trade-off', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=10)

    # Add annotation explaining the trade-off
    ax.text(0.95, 0.05,
            'Top-right is ideal:\nHigh recall + Low latency',
            transform=ax.transAxes,
            fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            verticalalignment='bottom',
            horizontalalignment='right')

    plt.tight_layout()
    output_path = f"{output_dir}/speed_vs_accuracy.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f" Saved: {output_path}")


plot_speed_vs_accuracy(metrics, OUTPUT_DIR)

# %% [markdown]
# ## Visualization 5: Index Build Time

# %% Plot Build Time
def plot_build_time(metrics, output_dir):
    """Bar chart comparing index build times."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter out methods with no build time
    methods = []
    build_times = []
    for i, method in enumerate(metrics['methods']):
        if metrics['build_time_seconds'][i] > 0:
            methods.append(method)
            build_times.append(metrics['build_time_seconds'][i])

    if not methods:
        print("⚠ No methods with build time data")
        return

    # Create bar chart
    bars = ax.bar(methods, build_times, color=['#3498db', '#2ecc71'])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        minutes = height / 60
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{minutes:.1f} min\n({height:.0f}s)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Build Time (seconds)', fontsize=12)
    ax.set_title('Index Build Time Comparison (1M vectors)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = f"{output_dir}/build_time_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f" Saved: {output_path}")


plot_build_time(metrics, OUTPUT_DIR)

# %% [markdown]
# ## Visualization 6: ef Parameter Sensitivity Comparison

# %% Plot ef Sensitivity
def plot_ef_sensitivity(reports, output_dir):
    """Compare ef parameter sensitivity across methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'hnswlib': '#3498db', 'faiss': '#2ecc71'}
    markers = {'hnswlib': 'o', 'faiss': 's'}

    for method_name, report_info in reports.items():
        data = report_info['data']

        # Skip if no ef_sensitivity data
        if 'ef_sensitivity' not in data:
            continue

        ef_data = data['ef_sensitivity']

        efs = [x['ef'] for x in ef_data]
        recalls = [x['recall@10'] for x in ef_data]
        qps_values = [x['qps'] for x in ef_data]

        color = colors.get(method_name, '#95a5a6')
        marker = markers.get(method_name, 'x')

        # Plot recall
        ax1.plot(efs, recalls, marker=marker, linewidth=2, markersize=8,
                label=method_name, color=color)

        # Plot QPS
        ax2.plot(efs, qps_values, marker=marker, linewidth=2, markersize=8,
                label=method_name, color=color)

    # Recall plot
    ax1.set_xlabel('efSearch Parameter', fontsize=12)
    ax1.set_ylabel('Recall@10', fontsize=12)
    ax1.set_title('efSearch vs Recall@10', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # QPS plot
    ax2.set_xlabel('efSearch Parameter', fontsize=12)
    ax2.set_ylabel('Queries Per Second (QPS)', fontsize=12)
    ax2.set_title('efSearch vs Search Throughput', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = f"{output_dir}/ef_sensitivity_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f" Saved: {output_path}")


plot_ef_sensitivity(reports, OUTPUT_DIR)

# %% [markdown]
# ## Generate Summary Report

# %% Summary Report
def generate_summary_report(metrics, reports, output_dir):
    """Generate a comprehensive text summary."""

    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("VECTOR SEARCH BENCHMARK COMPARISON")
    summary_lines.append("=" * 80)
    summary_lines.append("")

    # Dataset info (from first report)
    first_report = list(reports.values())[0]['data']
    metadata = first_report.get('metadata', {})

    summary_lines.append("DATASET:")
    summary_lines.append(f"  Dataset: {metadata.get('dataset', 'N/A')}")
    summary_lines.append(f"  Corpus size: {metadata.get('n_corpus_docs', 'N/A'):,} documents")
    summary_lines.append(f"  Queries: {metadata.get('n_queries', 'N/A')} test queries")
    summary_lines.append(f"  Embedding dimension: {metadata.get('embedding_dimension', 'N/A')}")
    summary_lines.append("")

    # Method comparison
    summary_lines.append("METHODS COMPARED:")
    summary_lines.append("")

    for i, method in enumerate(metrics['methods']):
        summary_lines.append(f"{i+1}. {method.upper()}")
        summary_lines.append(f"   Parameters: {metrics['params'][i]}")
        summary_lines.append(f"   Search latency: {metrics['search_latency_ms'][i]:.2f} ms/query")
        summary_lines.append(f"   Throughput: {metrics['queries_per_second'][i]:.1f} QPS")
        summary_lines.append(f"   Recall@10: {metrics['recall_at_10'][i]:.4f}")
        summary_lines.append(f"   MRR: {metrics['mrr'][i]:.4f}")

        if metrics['build_time_seconds'][i] > 0:
            build_min = metrics['build_time_seconds'][i] / 60
            summary_lines.append(f"   Build time: {build_min:.1f} minutes")

        summary_lines.append("")

    # Speedup analysis
    summary_lines.append("SPEEDUP ANALYSIS:")
    summary_lines.append("")

    if 'brute_force' in metrics['methods']:
        bf_idx = metrics['methods'].index('brute_force')
        bf_latency = metrics['search_latency_ms'][bf_idx]

        for i, method in enumerate(metrics['methods']):
            if method == 'brute_force':
                continue

            speedup = bf_latency / metrics['search_latency_ms'][i]
            recall_loss = metrics['recall_at_10'][bf_idx] - metrics['recall_at_10'][i]
            recall_pct = (1 - recall_loss / metrics['recall_at_10'][bf_idx]) * 100

            summary_lines.append(f"{method}:")
            summary_lines.append(f"  {speedup:.1f}x faster than brute force")
            summary_lines.append(f"  Recall: {recall_pct:.1f}% of brute force quality")
            summary_lines.append("")

    # Key findings
    summary_lines.append("KEY FINDINGS:")
    summary_lines.append("")

    # Find fastest method
    fastest_idx = metrics['search_latency_ms'].index(min(metrics['search_latency_ms']))
    summary_lines.append(f" Fastest search: {metrics['methods'][fastest_idx]} ({metrics['search_latency_ms'][fastest_idx]:.2f} ms/query)")

    # Find best recall
    best_recall_idx = metrics['recall_at_10'].index(max(metrics['recall_at_10']))
    summary_lines.append(f" Best recall: {metrics['methods'][best_recall_idx]} ({metrics['recall_at_10'][best_recall_idx]:.4f})")

    # Find best QPS
    best_qps_idx = metrics['queries_per_second'].index(max(metrics['queries_per_second']))
    summary_lines.append(f" Highest throughput: {metrics['methods'][best_qps_idx]} ({metrics['queries_per_second'][best_qps_idx]:.0f} QPS)")

    summary_lines.append("")
    summary_lines.append("=" * 80)

    # Print to console
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    # Save to file
    output_path = f"{output_dir}/BENCHMARK_SUMMARY.txt"
    with open(output_path, 'w') as f:
        f.write(summary_text)

    print(f"\n Saved: {output_path}")


generate_summary_report(metrics, reports, OUTPUT_DIR)

# %% [markdown]
# ## Summary
#
# This comparison reveals the classic **speed vs accuracy trade-off** in vector search:
#
# - **Brute Force**: 100% recall, but extremely slow (~700ms/query)
# - **HNSW methods**: 10-1000x faster with 75-80% recall
# - **FAISS**: Fastest search (0.08ms/query) with competitive recall
#
# **When to use each:**
# - **Brute Force**: Small datasets (<10K), need perfect recall
# - **HNSWlib**: Good balance, easy to tune, popular in production
# - **FAISS**: Maximum speed, production scale, GPU support

# %% Final Output
print(f"\n{'=' * 80}")
print(f"COMPARISON COMPLETE")
print(f"{'=' * 80}\n")
print(f"Generated visualizations:")
print(f"  - latency_comparison.png")
print(f"  - qps_comparison.png")
print(f"  - recall_comparison.png")
print(f"  - speed_vs_accuracy.png")
print(f"  - build_time_comparison.png")
print(f"  - ef_sensitivity_comparison.png")
print(f"  - BENCHMARK_SUMMARY.txt")
print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
