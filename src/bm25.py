import os
import pandas as pd
import numpy as np
import spacy
import pickle

from rank_bm25 import BM25Okapi
from src.utils import preprocess_spacy

def bm25_tokenize(text):
    """Tokenize text from query, including lowercase all letters, 
    removing number, space, or hyphen and whitespaces. 

    Parameters
    ----------
    text : str
        The text we want to tokenize from corpus.
    """
    nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])
    text = preprocess_spacy(nlp(text))
    return text.split()

def bm25_search(query, index_path, products_df, top_k=5):
    """Run BM25 search againts persisted BM25 index.

    Parameters
    ----------
    query : str
        User query text to embed and search.
    index_path : str
        The file path with the BM25 object
    products_df : pandas.DataFrame
        DataFrame aligned to index row positions. Must include
        ``product_title``, ``text``, and ``rating`` columns.
    top_k : int, default=5
        Number of nearest neighbors to retrieve.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of the top matches containing ``product_title``,
        ``text``, ``rating``, and ``score``, sorted by ``score`` in
        descending order.
    """
    with open(index_path, "rb") as f:
        bm25 = pickle.load(f)
    tokenized_query = bm25_tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    ranked_idx = np.argsort(scores)[::-1][:top_k]
    
    results = products_df.iloc[ranked_idx][['product_title', 'text', 'rating']]
    results['score'] = scores[ranked_idx]
    
    return results.sort_values('score', ascending=False)

def build_bm25(path, corpus):
    """Build BM25 indices and save the object to a given path.

    Parameters
    ----------
    path : str
        The file path where we want to save the BM25 object
    corpus : pd.DataFrame
        The corpus text for getting the products
    """
    if not os.path.exists(path):
        # tokenize corpus
        tokenized_products = [text.split() for text in corpus["text"]]
        bm25 = BM25Okapi(tokenized_products)
        
        # save to pickle
        with open(path, "wb") as f:
            pickle.dump(bm25, f)

    # load it
    with open(path, "rb") as f:
        bm25 = pickle.load(f)

    return bm25