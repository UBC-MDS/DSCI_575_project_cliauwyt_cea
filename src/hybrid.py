import spacy 
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bm25 import preprocess_spacy
from src.rag_pipeline import semantic_retriever


def preprocess_query(query) -> str:
    """
    Preprocess a user query using spaCy and a custom preprocessing function.

    This function loads a spaCy language model, processes the input query,
    and applies additional text normalization via `preprocess_spacy`.

    Parameters
    -------------
    query : str
        The raw user input query string.

    Returns
    -------------
    str
        The cleaned and preprocessed query string ready for retrieval.
    """
    nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])
    query = preprocess_spacy(nlp(query))
    return query


def bm25_retriever(docs, k=5):
    """Create a BM25 retriever from documents.

    Parameters
    ----------
    docs : list
        Documents used to build the BM25 index.
    k : int, default=5
        Number of top documents to return for each query.

    Returns
    -------
    BM25Retriever
        Configured BM25 retriever instance.
    """
    retriever = BM25Retriever.from_documents(
        docs,
        k=k
    )
    return retriever


def hybrid_retriever(bm25_retriever, vector_retriever):
    """Combine sparse and dense retrievers into an ensemble retriever.

    Parameters
    ----------
    bm25_retriever : BM25Retriever
        Sparse retriever based on BM25 scoring.
    vector_retriever : object
        Dense similarity retriever from a vector store.

    Returns
    -------
    EnsembleRetriever
        Weighted ensemble retriever that merges both retrieval strategies.
    """
    # Hybrid ensemble
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )
    return ensemble_retriever


if __name__ == "__main__":
    from dotenv import load_dotenv
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from vectorstore import csv_loader
    from rag_pipeline import load_llm, semantic_retriever, initialize_rag_chain
    from prompts import prompt

    load_dotenv()

    # Docs
    corpus_path = 'data/processed/preprocessed_corpus.csv'
    docs = csv_loader(corpus_path)

    # Embedding
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vectorstore
    vector_path = "data/processed/vector_store"
    vectorstore = FAISS.load_local(
        vector_path, embeddings, allow_dangerous_deserialization=True
    )

    # Retrievers
    ensemble_retriever = hybrid_retriever(
        bm25_retriever(docs, k=3),
        semantic_retriever(vectorstore, k=3)
    )

    # LLM
    llm = load_llm()

    # Defend against missing or blank CLI input.
    if len(sys.argv) < 2 or not any(arg.strip() for arg in sys.argv[1:]):
        print('Usage: python src/hybrid.py "<query>"')
        sys.exit(1)

    q = " ".join(sys.argv[1:]).strip()

    rag_chain = initialize_rag_chain(ensemble_retriever, llm, prompt)
    print("Answer:", rag_chain.invoke(q))
