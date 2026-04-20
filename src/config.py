from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_MERGED_PARQUET_PATH = RAW_DATA_DIR / "merged.parquet"
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "clean_data.csv"
PREPROCESSED_CORPUS_PATH = PROCESSED_DATA_DIR / "preprocessed_corpus.csv"
BM25_INDEX_PATH = PROCESSED_DATA_DIR / "bm25.pkl"
SEMANTIC_INDEX_PATH = PROCESSED_DATA_DIR / "embedding.faiss"
VECTOR_STORE_DIR = PROCESSED_DATA_DIR / "vector_store"
