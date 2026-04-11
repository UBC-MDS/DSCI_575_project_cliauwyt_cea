import pandas as pd
import re

def bm25_tokenize(text):
    """Tokenize text from corpus, including removing 
    lowercase letter, number, space, or hyphen and whitespaces. 

    Parameters
    ----------
    text : str
        The text we want to tokenize from corpus.
    """
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    return text.split()

def bm25_search(query, bm25, products_df, top_k=3):
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
    top_k : int, default=3
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
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(products_df[i], scores[i]) for i in ranked_idx]