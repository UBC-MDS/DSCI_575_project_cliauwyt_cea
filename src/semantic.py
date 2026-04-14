import faiss


def build_semantic_index(corpus, model, index_path):
    """Build and persist a FAISS L2 index from corpus text embeddings.

    Parameters
    ----------
    corpus : pandas.DataFrame
        DataFrame containing a ``text`` column with product
        descriptions/documents.
    model : sentence_transformers.SentenceTransformer
        Embedding model used to encode corpus text into dense vectors.
    index_path : str
        Destination path where the FAISS index will be written.
    """
    product_embeddings = model.encode(corpus.text.tolist())
    index = faiss.IndexFlatL2(product_embeddings.shape[1])
    index.add(product_embeddings)
    faiss.write_index(index, index_path)


def semantic_search(query, index_path, model, products_df, top_k=5):
    """Run semantic search against a persisted FAISS index.

    Parameters
    ----------
    query : str
        User query text to embed and search.
    index_path : str
        File path to a previously saved FAISS index.
    model : sentence_transformers.SentenceTransformer
        Embedding model used to encode the query.
    products_df : pandas.DataFrame
        DataFrame aligned to index row positions. Must include
        ``product_title``, ``text``, and ``rating`` columns.
    top_k : int, default=5
        Number of nearest neighbors to retrieve.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of the top matches containing ``product_title``,
        ``review_text``, ``rating``, and ``score``, sorted by ``score`` in
        descending order.
    """
    loaded_index = faiss.read_index(index_path)
    query_embedding = model.encode([query])
    scores, indices = loaded_index.search(query_embedding, top_k)
    results = products_df.iloc[indices[0]][['product_title', 'review_text', 'rating']]
    results['score'] = scores[0]
    return results.sort_values('score', ascending=False)
