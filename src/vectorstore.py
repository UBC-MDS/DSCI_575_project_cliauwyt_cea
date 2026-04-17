from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def csv_loader(
        corpus_path,
        content_columns=['text'],
        metadata_columns=['asin', 'product_title', 'rating', 'review_text']
        ):
    loader = CSVLoader(
        corpus_path,
        content_columns=content_columns,
        metadata_columns=metadata_columns,
        encoding='utf-8'
    )
    return loader.load()


def build_vectorstore(documents, vector_path, embeddings):
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(documents)

    # Compute embeddings and store in index
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(vector_path)


if __name__ == "__main__":
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    corpus_path = 'data/processed/preprocessed_corpus.csv'
    vector_path = "data/processed/vector_store"

    build_vectorstore(corpus_path, vector_path, embeddings)

    print(f"Saved vector store to {vector_path}")
