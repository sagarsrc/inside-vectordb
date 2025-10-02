# %% [markdown]
# # BEIR Dataset Download and Exploration
#
# This notebook downloads and showcases BEIR benchmark datasets for vector database evaluation.
# BEIR provides text corpus, queries, and ground truth relevance judgments (qrels).
#
# ## SciFact Dataset
#
# **Purpose**: Scientific claim verification - determine if scientific claims are supported by research papers
#
# **Dataset Structure**:
# - **Corpus**: ~5,183 scientific paper abstracts from biomedical domain
#   - Each document has: `doc_id`, `title`, `text`
# - **Queries**: 300 scientific claims to verify
#   - Each query is a claim statement (e.g., "Microstructural development of newborn cerebral white matter can be assessed in vivo")
# - **Qrels**: Ground truth relevance judgments
#   - Maps which papers support/refute each claim
#   - Relevance scores: 1 (relevant) or higher
#
# **Example Corpus Entry**:
# ```
# [Document 1] ID: 4983
# Title: Microstructural development of human newborn cerebral white matter assessed in vivo by diffusion tensor magnetic resonance imaging.
# Text: Alterations of the architecture of cerebral white matter in the developing human brain can affect cortical development and result in functional disabilities. A line scan diffusion-weighted magnetic re...
# ```
#
# **Fields Explanation**:

# - `doc_id` (str): Unique document identifier (e.g., "4983")
# - `title` (str): Paper title
# - `text` (str): Abstract text (full content for retrieval)
#
# **Use Case**: Test if vector DB can retrieve relevant scientific evidence for claims

# %% [markdown]
# ## Configuration

# %% Global Configuration
DATA_ROOT = "/Users/sagar/not-work/inside-vectordb-hnsw/data"
DATASET_NAME = "scifact"  # Small dataset: ~5K docs, 300 queries

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
        dataset_name: Name of BEIR dataset (e.g., 'scifact', 'nfcorpus')
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
def load_beir_data(data_path: str, split: str = "test"):
    """
    Load BEIR dataset components.

    Args:
        data_path: Path to dataset directory
        split: Dataset split ('train', 'dev', 'test')

    Returns:
        Tuple of (corpus, queries, qrels)
    """
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    return corpus, queries, qrels


corpus, queries, qrels = load_beir_data(dataset_path, split="test")

# %%
# corpus is a dictionary of corpus_id to a dictionary of title and text
# example: {'167944455': {'title': 'Microstructural development of human newborn cerebral white matter assessed in vivo by diffusion tensor magnetic resonance imaging.', 'text': 'Alterations of the architecture of cerebral white matter in the developing human brain can affect cortical development and result in functional disabilities. A line scan diffusion-weighted magnetic re...'}}

corpus["167944455"]["text"][:100]

# %%
# queries is a dictionary of query_id to a dictionary of query_text
# example: {'31715818': 1}
# '31715818' is the corpus_id
# 1 means relevant

queries["1"]
# %%
# qrels is a dictionary of query_id to a dictionary of corpus_id to relevance score
# example: {'31715818': {'31715818': 1}}
# '31715818' is the corpus_id
# 1 means relevant

qrels["1"]

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
