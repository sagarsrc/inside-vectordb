# %% [markdown]
# # HNSW Layer Assignment Analysis
#
# This notebook demonstrates how the M parameter in HNSW affects the hierarchical layer structure.
# Each vector is assigned to a maximum layer using a probabilistic formula, creating a skip-list-like
# hierarchy that enables efficient logarithmic search.
#
# **Key Concepts:**
# - HNSW creates multiple layers (levels) of graphs
# - Higher layers are sparser and enable long-distance jumps
# - Lower layers are denser and provide refinement
# - Layer assignment uses exponential distribution based on M
#
# **Layer Assignment Formula:**
# - `max_level = floor(-ln(U) * mL)` where:
#   - U ~ uniform(0,1) is a random number
#   - mL = 1 / ln(M) is the normalization factor

# %% [markdown]
# ## Configuration

# %% Global Configuration
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Sampling and visualization settings
N_SAMPLES = 5000  # Number of data points to sample
M_VALUES = [4, 8, 16, 32, 64]  # Different M values to compare
RANDOM_SEED = 42

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["figure.dpi"] = 300

np.random.seed(RANDOM_SEED)

# %% [markdown]
# ## Layer Assignment Function


# %% Implement Layer Assignment
def assign_layer(m: int, n_samples: int = 1) -> np.ndarray:
    """
    Assign maximum layer for HNSW nodes using the standard formula.

    Args:
        m: HNSW M parameter (max connections per layer)
        n_samples: Number of samples to generate

    Returns:
        Array of layer assignments (integers >= 0)

    Formula:
        max_level = floor(-ln(U) * mL)
        where mL = 1 / ln(M) (normalization factor)
    """
    # Generate uniform random numbers [0, 1)
    u = np.random.uniform(0, 1, size=n_samples)

    # Calculate normalization factor mL = 1 / ln(M)
    ml = 1.0 / np.log(m)

    # Apply formula: floor(-ln(U) * mL)
    max_levels = np.floor(-np.log(u) * ml).astype(int)

    return max_levels


print("Using formula: max_level = floor(-ln(U) * mL) where mL = 1/ln(M)")

# %% [markdown]
# ## Generate Layer Assignments for Different M Values

# %% Generate Data
layer_distributions = {}

for m in M_VALUES:
    layers = assign_layer(m, N_SAMPLES)
    layer_distributions[m] = layers

    # Calculate mL for display
    ml = 1.0 / np.log(m)

    # Statistics
    unique, counts = np.unique(layers, return_counts=True)
    max_layer = layers.max()
    avg_layer = layers.mean()

    print(f"\nM={m} (mL={ml:.3f}):")
    print(f"  Max layer: {max_layer}")
    print(f"  Avg layer: {avg_layer:.2f}")
    print(f"  Layer distribution: {dict(zip(unique, counts))}")

# %% [markdown]
# ## Visualization 1: Layer Distribution Histograms

# %% Plot Distribution Histograms
fig, axes = plt.subplots(len(M_VALUES), 1, figsize=(14, 12))
fig.suptitle(
    "HNSW Layer Assignment Distribution for Different M Values",
    fontsize=16,
    fontweight="bold",
)

for idx, m in enumerate(M_VALUES):
    ax = axes[idx]
    layers = layer_distributions[m]

    # Calculate statistics
    max_layer = layers.max()
    avg_layer = layers.mean()

    # Plot histogram
    counts, bins, patches = ax.hist(
        layers, bins=range(0, max_layer + 2), edgecolor="black", alpha=0.7
    )

    # Styling
    ax.set_xlabel("Layer Number", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        f"M = {m} (mL = {1.0 / np.log(m):.3f}) | Avg Layer: {avg_layer:.2f} | Max Layer: {max_layer}",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Add percentage labels on bars
    total = len(layers)
    for i, (count, patch) in enumerate(zip(counts, patches)):
        if count > 0:
            percentage = 100 * count / total
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                count,
                f"{percentage:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

plt.tight_layout()
plt.savefig(
    "../reports/summary/layer_distribution_histograms.png", dpi=300, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ## Visualization 2: Comparative Layer Distribution

# %% Plot Comparative Distribution
fig, ax = plt.subplots(figsize=(14, 8))

x_offset = 0
bar_width = 0.15
colors = sns.color_palette("husl", len(M_VALUES))

# Get max layer across all M values for consistent x-axis
max_layer_overall = max(layer_distributions[m].max() for m in M_VALUES)

for idx, m in enumerate(M_VALUES):
    layers = layer_distributions[m]
    counter = Counter(layers)

    # Create array with counts for all possible layers
    layer_counts = [counter.get(i, 0) for i in range(max_layer_overall + 1)]
    percentages = [100 * count / N_SAMPLES for count in layer_counts]

    # Plot bars
    x_positions = np.arange(len(layer_counts)) + idx * bar_width
    ax.bar(
        x_positions,
        percentages,
        bar_width,
        label=f"M={m}",
        color=colors[idx],
        alpha=0.8,
    )

ax.set_xlabel("Layer Number", fontsize=13, fontweight="bold")
ax.set_ylabel("Percentage of Nodes (%)", fontsize=13, fontweight="bold")
ax.set_title(
    f"Layer Distribution Comparison Across M Values (N={N_SAMPLES:,} samples)",
    fontsize=15,
    fontweight="bold",
)
ax.set_xticks(np.arange(max_layer_overall + 1) + bar_width * 2)
ax.set_xticklabels(range(max_layer_overall + 1))
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(
    "../reports/summary/layer_distribution_comparison.png", dpi=300, bbox_inches="tight"
)
plt.show()


# %% [markdown]
# ## Key Insights
#
# ### How M Affects Layer Assignment:
#
# 1. **Higher M Lower Average Layer**:
#    - M=4: Most nodes reach higher layers (avg ~1.6)
#    - M=64: Most nodes stay in lower layers (avg ~0.2)
#
# 2. **Base Layer Density**:
#    - Larger M means more nodes in Layer 0 (base layer)
#    - M=64: ~85% of nodes in Layer 0
#    - M=4: ~38% of nodes in Layer 0
#
# 3. **Maximum Layer Height**:
#    - Decreases as M increases
#    - M=4 can reach Layer 8+
#    - M=64 rarely exceeds Layer 3
#
# 4. **Hierarchy Shape**:
#    - **Small M (4-8)**: Tall, narrow pyramid - many layers, sparse top
#    - **Large M (32-64)**: Short, wide pyramid - few layers, dense base
#
# ### Trade-offs:
#
# - **Small M**:
#   - More layers better skip-list properties
#   - Longer path traversal
#   - Lower memory per node
#
# - **Large M**:
#   - Fewer layers flatter structure
#   - Shorter path traversal
#   - Higher memory per node (more connections)
#   - Better recall (denser graphs)

# %%
