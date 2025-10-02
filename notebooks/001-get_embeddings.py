# %% [markdown]
# # Generate Embeddings for BEIR Dataset
#
# This notebook generates vector embeddings for BEIR benchmark corpus and queries
# using sentence-transformers. Embeddings are stored in compressed `.npz` format.
#
# - `.npz`: Stores multiple arrays (compressed) - **saves ~50-70% space**
# - Works with any BEIR dataset (MS MARCO, SciFact, NFCorpus, etc.)
# - **GPU accelerated**: Automatically uses CUDA if available

# **Output**:
# - `{dataset_name}_corpus_embeddings.npz`: Document embeddings + doc IDs
# - `{dataset_name}_query_embeddings.npz`: Query embeddings + query IDs

# %% [markdown]
# ## Configuration

# %% Global Configuration
DATA_ROOT = "../data"
DATASET_NAME = "msmarco"  # Can be any BEIR dataset: msmarco, scifact, nfcorpus, etc.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast, good quality
BATCH_SIZE = 2048  # Increased for GPU - use 32 for CPU
SPLIT = "dev"  # For MS MARCO: 'dev' (6,980 queries) or 'test' (43 queries)
USE_SUBSET = True  # Use 1M subset for faster experimentation (8.8x faster)
SUBSET_SIZE = "1M"  # Which subset to use (must exist in subsets/ folder)

# %% [markdown]
# ## Import Dependencies

# %% Imports
import os
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader
import dotenv

dotenv.load_dotenv()

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("Running on CPU")

# %% [markdown]
# ## Load Dataset


# %% Load BEIR Dataset
def load_beir_dataset(
    data_root: str,
    dataset_name: str,
    split: str = "dev",
    use_subset: bool = False,
    subset_size: str = "1M",
):
    """
    Load any BEIR dataset (generic function).

    Args:
        data_root: Root directory containing datasets
        dataset_name: Name of dataset (e.g., 'msmarco', 'scifact', 'nfcorpus')
        split: Dataset split to load ('train', 'dev', 'test')
        use_subset: Whether to use a corpus subset from subsets/ folder
        subset_size: Which subset to use (e.g., '1M')

    Returns:
        Tuple of (corpus, queries, qrels)
    """
    import json

    dataset_path = os.path.join(data_root, dataset_name)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Run 000-get_data.py first to download the dataset."
        )

    print(f"Loading {dataset_name} dataset ({split} split)...")

    if use_subset:
        # Load corpus from subset file
        subset_path = os.path.join(
            dataset_path, "subsets", f"corpus_{subset_size}.jsonl"
        )
        if not os.path.exists(subset_path):
            raise FileNotFoundError(
                f"Subset not found at {subset_path}. "
                f"Run 000-get_data.py to create the subset first."
            )

        print(f"Loading corpus subset from: {subset_path}")
        corpus = {}
        with open(subset_path, "r") as f:
            for line in f:
                doc = json.loads(line)
                corpus[doc["_id"]] = {
                    "title": doc.get("title", ""),
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {}),
                }

        # Load queries and qrels directly from files (without loading full corpus)
        queries_file = os.path.join(dataset_path, "queries.jsonl")
        qrels_file = os.path.join(dataset_path, "qrels", f"{split}.tsv")

        queries = {}
        with open(queries_file, "r") as f:
            for line in f:
                query = json.loads(line)
                queries[query["_id"]] = query["text"]

        qrels = {}
        with open(qrels_file, "r") as f:
            next(f)  # Skip header
            for line in f:
                query_id, doc_id, score = line.strip().split("\t")
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = int(score)
    else:
        # Load full dataset normally
        corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path).load(
            split=split
        )

    print(f"Loaded {dataset_name} - {split} split:")
    print(f"  Corpus: {len(corpus):,} documents {'(SUBSET)' if use_subset else ''}")
    print(f"  Queries: {len(queries):,} queries")
    print(f"  Qrels: {sum(len(docs) for docs in qrels.values()):,} relevance judgments")

    return corpus, queries, qrels


corpus, queries, qrels = load_beir_dataset(
    DATA_ROOT, DATASET_NAME, split=SPLIT, use_subset=USE_SUBSET, subset_size=SUBSET_SIZE
)

# %% [markdown]
# ## Load Embedding Model


# %% Load Sentence Transformer Model
def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Load sentence transformer model for embedding generation.
    Automatically uses GPU if available.

    Args:
        model_name: HuggingFace model name

    Returns:
        SentenceTransformer model
    """
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # Model automatically uses CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model loaded on: {device}")
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

    return model


model = load_embedding_model(EMBEDDING_MODEL)

# %% [markdown]
# ## Generate Corpus Embeddings


# %% Generate and Save Corpus Embeddings
def generate_corpus_embeddings(
    corpus: dict, model: SentenceTransformer, batch_size: int = 32
) -> tuple[np.ndarray, list[str]]:
    """
    Generate embeddings for all documents in corpus.

    Args:
        corpus: Dictionary of {doc_id: {'title': str, 'text': str}}
        model: SentenceTransformer model
        batch_size: Batch size for encoding

    Returns:
        Tuple of (embeddings array, list of doc_ids)
    """
    doc_ids = list(corpus.keys())

    # Combine title + text for better embeddings
    texts = [
        f"{corpus[doc_id].get('title', '')} {corpus[doc_id]['text']}"
        for doc_id in doc_ids
    ]

    print(f"\nGenerating embeddings for {len(texts):,} documents...")
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
    )

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings, doc_ids


corpus_embeddings, corpus_ids = generate_corpus_embeddings(corpus, model, BATCH_SIZE)

# %% [markdown]
# ## Generate Query Embeddings


# %% Generate and Save Query Embeddings
def generate_query_embeddings(
    queries: dict, model: SentenceTransformer, batch_size: int = 4096
) -> tuple[np.ndarray, list[str]]:
    """
    Generate embeddings for all queries.

    Args:
        queries: Dictionary of {query_id: query_text}
        model: SentenceTransformer model
        batch_size: Batch size for encoding

    Returns:
        Tuple of (embeddings array, list of query_ids)
    """
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    print(f"\nGenerating embeddings for {len(query_texts):,} queries...")
    embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings, query_ids


query_embeddings, query_ids = generate_query_embeddings(queries, model, BATCH_SIZE)

# %% [markdown]
# ## Save Embeddings to Disk


# %% Save Embeddings as NPZ
def save_embeddings_npz(embeddings: np.ndarray, ids: list[str], output_path: str):
    """
    Save embeddings and IDs to compressed NPZ file.

    Args:
        embeddings: Numpy array of embeddings
        ids: List of IDs corresponding to embeddings
        output_path: Path to save NPZ file
    """
    # Convert IDs to numpy array
    ids_array = np.array(ids, dtype=object)

    # Save with compression
    np.savez_compressed(output_path, embeddings=embeddings, ids=ids_array)

    # Check file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved to: {output_path}")
    print(f"File size: {file_size_mb:.2f} MB")

    # Show space savings
    uncompressed_size = embeddings.nbytes / (1024 * 1024)
    compression_ratio = (1 - file_size_mb / uncompressed_size) * 100
    print(f"Compression: {compression_ratio:.1f}% space saved")


# Save corpus embeddings
subset_suffix = f"_{SUBSET_SIZE}" if USE_SUBSET else ""
corpus_embeddings_path = os.path.join(
    DATA_ROOT, f"{DATASET_NAME}{subset_suffix}_corpus_embeddings.npz"
)
save_embeddings_npz(corpus_embeddings, corpus_ids, corpus_embeddings_path)

# Save query embeddings
query_embeddings_path = os.path.join(
    DATA_ROOT, f"{DATASET_NAME}{subset_suffix}_query_embeddings.npz"
)
save_embeddings_npz(query_embeddings, query_ids, query_embeddings_path)

# %% [markdown]
# ## Verify Saved Embeddings


# %% Load and Verify Embeddings
def load_and_verify_embeddings(npz_path: str):
    """
    Load and display information about saved embeddings.

    Args:
        npz_path: Path to NPZ file
    """
    print(f"\nLoading: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    embeddings = data["embeddings"]
    ids = data["ids"]

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of IDs: {len(ids)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"First 3 IDs: {ids[:3]}")
    print(f"Sample embedding (first 10 dims): {embeddings[0][:10]}")

    return embeddings, ids


print("=" * 80)
print("CORPUS EMBEDDINGS")
print("=" * 80)
corpus_emb_loaded, corpus_ids_loaded = load_and_verify_embeddings(
    corpus_embeddings_path
)

print("\n" + "=" * 80)
print("QUERY EMBEDDINGS")
print("=" * 80)
query_emb_loaded, query_ids_loaded = load_and_verify_embeddings(query_embeddings_path)

# %% [markdown]
# ## Summary


# %% Display Summary
def display_summary():
    """Display summary of generated embeddings."""
    print("\n" + "=" * 80)
    print("EMBEDDING GENERATION SUMMARY")
    print("=" * 80)
    print(f"\nModel: {EMBEDDING_MODEL}")
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print(f"\nCorpus embeddings: {corpus_embeddings.shape[0]:,} documents")
    print(f"Query embeddings: {query_embeddings.shape[0]:,} queries")
    print(f"\nFiles saved:")
    print(f"  - {corpus_embeddings_path}")
    print(f"  - {query_embeddings_path}")


display_summary()

#%% [markdown]
# Expected statistics after running 001-get_embeddings.py
# Model: sentence-transformers/all-MiniLM-L6-v2
# Embedding dimension: 384

# Corpus embeddings: 1,000,000 documents
# Query embeddings: 6,980 queries

# Files saved:
#   - ../data/msmarco_1M_corpus_embeddings.npz
#   - ../data/msmarco_1M_query_embeddings.npz