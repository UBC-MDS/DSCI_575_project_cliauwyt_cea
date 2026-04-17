from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def csv_loader(
        corpus_path,
        content_columns=['text'],
        metadata_columns=['asin', 'product_title', 'rating', 'review_text']
        ):
    """Load review corpus documents from a CSV file.

    Parameters
    ----------
    corpus_path : str
        Path to the input CSV file.
    content_columns : list of str, default=['text']
        Column names used as document content.
    metadata_columns : list of str,
    default=['asin', 'product_title', 'rating', 'review_text']
        Column names stored as document metadata.

    Returns
    -------
    list
        List of loaded document objects.
    """
    loader = CSVLoader(
        corpus_path,
        content_columns=content_columns,
        metadata_columns=metadata_columns,
        encoding='utf-8'
    )
    return loader.load()


def build_vectorstore(documents, vector_path, embeddings):
    """Split documents, build a FAISS index, and save it locally.

    Parameters
    ----------
    documents : list
        Input documents to index.
    vector_path : str
        Output directory where the FAISS vector store is saved.
    embeddings : object
        Embedding model used to convert document chunks into vectors.

    Returns
    -------
    None
        This function saves the vector store to disk and does not
        return a value.
    """
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

    # Initialize the sentence-transformer model used for embeddings.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Define input corpus location and output vector store directory.
    corpus_path = 'data/processed/preprocessed_corpus.csv'
    vector_path = "data/processed/vector_store"

    # Build the FAISS index and persist it to disk.
    build_vectorstore(corpus_path, vector_path, embeddings)

    print(f"Saved vector store to {vector_path}")
