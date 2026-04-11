import faiss


def build_semantic_index(corpus, model, index_path):
    product_embeddings = model.encode(corpus.text.tolist())
    index = faiss.IndexFlatL2(product_embeddings.shape[1])
    index.add(product_embeddings)
    faiss.write_index(index, index_path)


def semantic_search(query, index_path, model, products_series, top_k=5):
    loaded_index = faiss.read_index(index_path)
    query_embedding = model.encode([query])
    scores, indices = loaded_index.search(query_embedding, top_k)
    return [(products_series.iloc[i], score) for score, i in zip(scores[0], indices[0])][::-1]
