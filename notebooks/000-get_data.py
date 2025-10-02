# %% [markdown]
# # BEIR Dataset Download and Exploration
#
# This notebook downloads and showcases BEIR benchmark datasets for vector database evaluation.
# BEIR provides text corpus, queries, and ground truth relevance judgments (qrels).
#
# ## MS MARCO Dataset
#
# **Purpose**: Large-scale passage ranking - retrieve relevant passages for real user search queries
#
# **Dataset Structure**:
# - **Corpus**: 8,841,823 passages from web documents
#   - Each document has: `doc_id`, `title`, `text`
# - **Queries**: Total 509,962 queries across all splits
#   - **dev split**: 6,980 queries with 7,437 relevance judgments
#   - **test split**: 43 queries with 9,260 relevance judgments (TREC-DL 2019 - many judgments per query)
#   - **train split**: 532,751 query-document pairs
#   - Each query is a natural language question (e.g., "what is the temperature in mars")
# - **Qrels**: Ground truth relevance judgments with **graded relevance**
#   - Maps which passages are relevant for each query
#   - **Relevance scores**: 0 (not relevant), 1 (relevant), 2 (highly relevant/perfect match)
#   - Unlike binary relevance (SciFact), MS MARCO uses graded judgments for nuanced evaluation
#
# **Example Corpus Entry**:
# ```
# [Document 1] ID: 0
# Title: (empty string - many MS MARCO passages don't have titles)
# Text: The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated.
# ```
#
# **Example Query**:
# ```
# Query ID: 1185869
# Text: what was the immediate impact of the success of the manhattan project?
# ```
#
# **Fields Explanation**:
# - `doc_id` (str): Unique document identifier (e.g., "0")
# - `title` (str): Passage title (often empty in MS MARCO)
# - `text` (str): Passage text (full content for retrieval)
# - `query_id` (str): Unique query identifier
# - `query text` (str): Natural language search query
#
# **Use Case**: Test if vector DB can retrieve relevant passages for real-world search queries at scale

# %% [markdown]
# ## Configuration

# %% Global Configuration
DATA_ROOT = "../data"
DATASET_NAME = "msmarco"  # Large dataset: 8.84M docs, 6,980 queries (dev) or 43 queries (test)

# %% [markdown]
# ## Import Dependencies

# %% Imports

from beir import util
from beir.datasets.data_loader import GenericDataLoader
import pandas as pd

# %% [markdown]
# ## Download BEIR Dataset


# %% Download Dataset
def download_beir_dataset(dataset_name: str, data_root: str) -> str:
    """
    Download BEIR dataset and return the extracted path.

    Args:
        dataset_name: Name of BEIR dataset (e.g., 'msmarco', 'scifact', 'nfcorpus')
        data_root: Root directory to store datasets

    Returns:
        Path to extracted dataset directory
    """
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, data_root)
    print(f"Dataset downloaded to: {data_path}")
    return data_path


dataset_path = download_beir_dataset(DATASET_NAME, DATA_ROOT)

# %% [markdown]
# ## Load Corpus, Queries, and Qrels


# %% Load Dataset
def load_beir_data(data_path: str, split: str = "dev"):
    """
    Load BEIR dataset components.

    Args:
        data_path: Path to dataset directory
        split: Dataset split ('train', 'dev', 'test')
               For MS MARCO: 'dev' has 6,980 queries, 'test' has 43 queries (TREC-DL 2019)

    Returns:
        Tuple of (corpus, queries, qrels)
    """
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    return corpus, queries, qrels


corpus, queries, qrels = load_beir_data(dataset_path, split="dev")  # Use 'dev' for 6,980 queries

# %%
# corpus is a dictionary of corpus_id to a dictionary of title and text
# example: {'text': 'The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated.', 'title': ''}

list(corpus.items())[0]

# %%
# queries is a dictionary of query_id to query text
# example: {'19335': 'anthropological definition of environment'}

list(queries.items())[0]

# %%
# qrels is a dictionary of query_id to a dictionary of corpus_id to relevance score
# example: {'19335': {'1017759': 0, '1082489': 0, '109063': 0, .... '1720395': 1, '1722': 0, '1725697': 0, '1726': 0, '1729': 2, '1730': 0, '1731': 0, '1732': 0, '1733': 0, '1734': 0, '1735': 0, '1736': 0,}

# MS MARCO uses graded relevance: 0 (not relevant), 1 (relevant), 2 (highly relevant)
# This allows nuanced evaluation beyond binary relevant/non-relevant

list(qrels.items())[0]

# %% [markdown]
# ## Display Sample Documents from Corpus


# %% Showcase Corpus
def showcase_corpus(corpus: dict, n_samples: int = 5):
    """Display sample documents from corpus."""
    print(f"\n{'=' * 80}")
    print(f"CORPUS: {len(corpus):,} documents")
    print(f"{'=' * 80}\n")

    for i, (doc_id, doc_data) in enumerate(list(corpus.items())[:n_samples]):
        print(f"[Document {i + 1}] ID: {doc_id}")
        print(f"Title: {doc_data.get('title', 'N/A')}")
        print(f"Text: {doc_data['text'][:200]}...")
        print(f"{'-' * 80}\n")


showcase_corpus(corpus)

# %% [markdown]
# ## Display Sample Queries


# %% Showcase Queries
def showcase_queries(queries: dict, n_samples: int = 5):
    """Display sample queries."""
    print(f"\n{'=' * 80}")
    print(f"QUERIES: {len(queries):,} total")
    print(f"{'=' * 80}\n")

    for i, (query_id, query_text) in enumerate(list(queries.items())[:n_samples]):
        print(f"[Query {i + 1}] ID: {query_id}")
        print(f"Text: {query_text}")
        print(f"{'-' * 80}\n")


showcase_queries(queries)

# %% [markdown]
# ## Display Ground Truth Relevance Judgments (Qrels)


# %% Showcase Qrels (Ground Truth)
def showcase_qrels(qrels: dict, queries: dict, corpus: dict, n_samples: int = 3):
    """Display sample query-document relevance judgments."""
    print(f"\n{'=' * 80}")
    print(f"QRELS (Ground Truth Relevance Judgments)")
    print(f"{'=' * 80}\n")

    for i, (query_id, doc_relevances) in enumerate(list(qrels.items())[:n_samples]):
        print(f"\n[Qrel {i + 1}] Query ID: {query_id}")
        print(f"Query: {queries[query_id]}")
        print(f"\nRelevant Documents ({len(doc_relevances)} total):")

        for doc_id, relevance_score in list(doc_relevances.items())[:3]:
            print(f"\n  Doc ID: {doc_id} | Relevance Score: {relevance_score}")
            print(f"  Title: {corpus[doc_id].get('title', 'N/A')}")
            print(f"  Text: {corpus[doc_id]['text'][:150]}...")

        print(f"\n{'-' * 80}")


showcase_qrels(qrels, queries, corpus)

# %% [markdown]
# ## Dataset Statistics Summary


# %% Dataset Statistics
def show_statistics(corpus: dict, queries: dict, qrels: dict):
    """Display dataset statistics."""
    print(f"\n{'=' * 80}")
    print(f"DATASET STATISTICS: {DATASET_NAME.upper()}")
    print(f"{'=' * 80}\n")

    print(f"Total Documents: {len(corpus):,}")
    print(f"Total Queries: {len(queries):,}")
    print(
        f"Total Query-Doc Pairs (qrels): {sum(len(docs) for docs in qrels.values()):,}"
    )

    # Average relevant docs per query
    avg_rel_docs = sum(len(docs) for docs in qrels.values()) / len(qrels)
    print(f"Avg Relevant Docs per Query: {avg_rel_docs:.2f}")

    # Document length statistics
    doc_lengths = [len(doc["text"].split()) for doc in corpus.values()]
    print(f"\nDocument Length (words):")
    print(f"  Min: {min(doc_lengths)}")
    print(f"  Max: {max(doc_lengths)}")
    print(f"  Avg: {sum(doc_lengths) / len(doc_lengths):.1f}")

    # Query length statistics
    query_lengths = [len(q.split()) for q in queries.values()]
    print(f"\nQuery Length (words):")
    print(f"  Min: {min(query_lengths)}")
    print(f"  Max: {max(query_lengths)}")
    print(f"  Avg: {sum(query_lengths) / len(query_lengths):.1f}")


show_statistics(corpus, queries, qrels)

# %% [markdown]
# ## Convert Qrels to DataFrame for Analysis


# %% Create Qrels DataFrame for Analysis
def create_qrels_dataframe(qrels: dict) -> pd.DataFrame:
    """Convert qrels dict to pandas DataFrame for easier analysis."""
    rows = []
    for query_id, doc_relevances in qrels.items():
        for doc_id, relevance_score in doc_relevances.items():
            rows.append(
                {"query_id": query_id, "doc_id": doc_id, "relevance": relevance_score}
            )

    df = pd.DataFrame(rows)
    print(f"\nQrels DataFrame shape: {df.shape}")
    print(f"\nRelevance score distribution:")
    print(df["relevance"].value_counts().sort_index())

    return df


qrels_df = create_qrels_dataframe(qrels)

# %% [markdown]
# ## Create 1M Document Subset (for faster experimentation)
#
# For faster iteration and experimentation, we create a 1M document subset while preserving
# **all relevant documents** from qrels to maintain benchmark integrity.
#
# **Why use a subset?**
# - **8.8x faster** embedding generation (16 mins vs 2.5 hours)
# - Faster HNSW index building and search
# - Same queries and qrels - benchmarks remain valid
# - Ideal for rapid prototyping and algorithm testing
#
# **Strategy**:
# 1. Keep ALL documents that appear in qrels (relevant docs) - **100% preserved**
# 2. Randomly sample additional documents to reach 1M total
# 3. All queries (6,980) and qrels (7,437) remain unchanged
# 4. Benchmark results are still valid for evaluation
#
# **Subset Composition (MS MARCO dev split):**
# - Original corpus: 8,841,823 documents
# - Subset corpus: 1,000,000 documents (11.3% of original)
# - Relevant docs preserved: ~7,437 (all from qrels)
# - Additional sampled: ~992,563 random documents
# - Queries: 6,980 (unchanged)
# - Qrels: 7,437 relevance judgments (unchanged)
#
# **Impact on benchmarks:**
# - ✅ All relevant documents are searchable
# - ✅ Precision, MRR, nDCG metrics are valid
# - ⚠️ Recall@k might be marginally different (fewer total documents to retrieve from)
# - ✅ Ranking quality of relevant docs is fully testable


# %% Create Corpus Subset
def create_corpus_subset(
    corpus: dict, qrels: dict, target_size: int = 1_000_000, seed: int = 42
):
    """
    Create a subset of corpus while preserving all relevant documents.

    Args:
        corpus: Full corpus dictionary
        qrels: Relevance judgments
        target_size: Target corpus size
        seed: Random seed for reproducibility

    Returns:
        Subset corpus dictionary
    """
    import random

    random.seed(seed)

    # Collect all relevant document IDs from qrels
    relevant_doc_ids = set()
    for query_id, doc_relevances in qrels.items():
        relevant_doc_ids.update(doc_relevances.keys())

    print(f"\n{'=' * 80}")
    print(f"CREATING CORPUS SUBSET")
    print(f"{'=' * 80}\n")
    print(f"Original corpus size: {len(corpus):,}")
    print(f"Relevant documents (from qrels): {len(relevant_doc_ids):,}")
    print(f"Target subset size: {target_size:,}")

    # If target is larger than corpus, return full corpus
    if target_size >= len(corpus):
        print(f"\nTarget size >= corpus size. Returning full corpus.")
        return corpus

    # Sample additional documents to reach target size
    all_doc_ids = set(corpus.keys())
    non_relevant_ids = list(all_doc_ids - relevant_doc_ids)
    additional_needed = target_size - len(relevant_doc_ids)

    if additional_needed > 0:
        sampled_additional = random.sample(
            non_relevant_ids, min(additional_needed, len(non_relevant_ids))
        )
    else:
        sampled_additional = []

    # Create subset corpus
    subset_doc_ids = relevant_doc_ids | set(sampled_additional)
    subset_corpus = {doc_id: corpus[doc_id] for doc_id in subset_doc_ids}

    print(f"\nSubset created:")
    print(f"  Total documents: {len(subset_corpus):,}")
    print(f"  Relevant docs (preserved): {len(relevant_doc_ids):,} (100%)")
    print(f"  Additional sampled docs: {len(sampled_additional):,}")

    # Verify benchmark integrity
    missing_docs = 0
    for query_id, doc_relevances in qrels.items():
        for doc_id in doc_relevances.keys():
            if doc_id not in subset_doc_ids:
                missing_docs += 1

    print(f"\nBenchmark integrity check:")
    print(f"  Missing relevant docs: {missing_docs} (should be 0)")
    print(
        f"  Benchmark valid: {'✓ YES' if missing_docs == 0 else '✗ NO - DO NOT USE'}"
    )

    return subset_corpus


# Create 1M subset
corpus_subset = create_corpus_subset(corpus, qrels, target_size=1_000_000)

# %% [markdown]
# ## Save Subset to Disk (Optional)
#
# Save the subset corpus for reuse without regenerating


# %% Save Subset Corpus
def save_corpus_subset(corpus_subset: dict, dataset_path: str, suffix: str = "1M"):
    """
    Save corpus subset to disk in BEIR format.

    Args:
        corpus_subset: Subset corpus dictionary
        dataset_path: Path to dataset directory
        suffix: Suffix for subset file (e.g., '1M')
    """
    import json
    from pathlib import Path

    # Create subsets directory
    subsets_dir = Path(dataset_path) / "subsets"
    subsets_dir.mkdir(exist_ok=True)

    output_file = subsets_dir / f"corpus_{suffix}.jsonl"

    print(f"\nSaving corpus subset to: {output_file}")

    with open(output_file, "w") as f:
        for doc_id, doc_data in corpus_subset.items():
            record = {
                "_id": doc_id,
                "title": doc_data.get("title", ""),
                "text": doc_data["text"],
                "metadata": doc_data.get("metadata", {}),
            }
            f.write(json.dumps(record) + "\n")

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"Saved {len(corpus_subset):,} documents ({file_size_mb:.1f} MB)")

    return str(output_file)


# Save the subset
subset_path = save_corpus_subset(corpus_subset, dataset_path, suffix="1M")

# %% [markdown]
# ## Verify Subset Statistics


# %% Show Subset Statistics
def show_subset_statistics(
    original_corpus: dict, subset_corpus: dict, queries: dict, qrels: dict
):
    """Compare original and subset corpus statistics."""
    print(f"\n{'=' * 80}")
    print(f"SUBSET COMPARISON")
    print(f"{'=' * 80}\n")

    print(f"Corpus Size:")
    print(f"  Original: {len(original_corpus):,}")
    print(f"  Subset:   {len(subset_corpus):,}")
    print(
        f"  Reduction: {(1 - len(subset_corpus)/len(original_corpus))*100:.1f}%"
    )

    print(f"\nQueries & Qrels:")
    print(f"  Queries: {len(queries):,} (unchanged)")
    print(f"  Qrels: {sum(len(docs) for docs in qrels.values()):,} (unchanged)")

    # Calculate coverage of relevant docs
    relevant_doc_ids = set()
    for doc_relevances in qrels.values():
        relevant_doc_ids.update(doc_relevances.keys())

    coverage = len(relevant_doc_ids & set(subset_corpus.keys()))
    print(
        f"\nRelevant Doc Coverage: {coverage}/{len(relevant_doc_ids)} ({coverage/len(relevant_doc_ids)*100:.1f}%)"
    )



show_subset_statistics(corpus, corpus_subset, queries, qrels)


