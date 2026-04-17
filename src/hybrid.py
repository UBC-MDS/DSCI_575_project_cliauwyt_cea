import spacy 
from src.bm25 import preprocess_spacy
from src.rag_pipeline import semantic_retriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


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
