import pandas as pd
import re
import numpy as np
import spacy
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

def bm25_search(query, bm25, products_df, top_k=5):
    """Run BM25 search againts persisted BM25 index.

    Parameters
    ----------
    query : str
        User query text to embed and search.
    bm25: BM25 index object
        The BM25 index object from tokenized corpus
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
    tokenized_query = bm25_tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    ranked_idx = np.argsort(scores)[::-1][:top_k]
    
    results = products_df.iloc[ranked_idx][['product_title', 'text', 'rating']]
    results['score'] = scores[ranked_idx]
    
    return results.sort_values('score', ascending=False)