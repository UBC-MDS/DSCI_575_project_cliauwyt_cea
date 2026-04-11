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


def semantic_search(query, index_path, model, products_series, top_k=5):
    """Run semantic search against a persisted FAISS index.

    Parameters
    ----------
    query : str
        User query text to embed and search.
    index_path : str
        File path to a previously saved FAISS index.
    model : sentence_transformers.SentenceTransformer
        Embedding model used to encode the query.
    products_series : pandas.Series
        Series containing product/document text aligned to index row positions.
    top_k : int, default=5
        Number of nearest neighbors to retrieve.

    Returns
    -------
    list[tuple[str, float]]
        List of ``(product_text, distance_score)`` pairs in reverse
        ranked order.
    """
    loaded_index = faiss.read_index(index_path)
    query_embedding = model.encode([query])
    scores, indices = loaded_index.search(query_embedding, top_k)
    return [
        (products_series.iloc[i], score)
        for score, i in zip(scores[0], indices[0])
    ][::-1]
