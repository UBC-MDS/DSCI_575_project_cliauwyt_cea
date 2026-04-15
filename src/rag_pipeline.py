from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("""You are a helpful Amazon shopping assistant.
    Answer the question using ONLY the following context (real product reviews + metadata).
    Always cite the product ASIN when possible.

    Customer Reviews: {context}

    Question: {query}

    Answer based on the reviews above:""")


def build_vectorstore(
        corpus_path, vector_path, embeddings,
        content_columns=['text'],
        metadata_columns=['asin', 'product_title', 'rating', 'review_text']
):
    # Load documents
    loader = CSVLoader(
        corpus_path,
        content_columns=content_columns,
        metadata_columns=metadata_columns,
        encoding='utf-8'
    )
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(documents)

    # Compute embeddings and store in index
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(vector_path)


def build_context(docs):
    return "\n\n".join(
        f"Product ASIN: {doc.metadata.get('asin', 'N/A')}\n"
        f"Title: {doc.metadata.get('product_title', 'N/A')}\n"
        f"Rating: {doc.metadata.get('rating', 'N/A')}/5.0\n"
        f"Review: {doc.metadata.get('review_text', 'N/A')}\n"
        for doc in docs
    )


def rag_pipeline(vectorstore, query, generator, k=5, prompt=prompt):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}  # Fetch k most similar documents
    )
    llm = HuggingFacePipeline(pipeline=generator)

    rag_chain = (
        {
            "context": retriever | build_context,
            "query": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(query)
    return answer

if __name__ == "__main__":
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    corpus_path = 'data/processed/preprocessed_corpus.csv'
    vector_path = "data/processed/vector_store"

    build_vectorstore(corpus_path, vector_path, embeddings)

    print(f"Saved vector store to {vector_path}")

