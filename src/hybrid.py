import spacy 
from src.bm25 import preprocess_spacy
from src.rag_pipeline import build_context
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
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


def initialize_hybrid_rag(docs, vectorstore, generator, prompt, k=5):
    """
    Initialize a hybrid Retrieval-Augmented Generation (RAG) pipeline.

    This function constructs a hybrid retriever that combines BM25 keyword
    search and semantic vector search, and integrates it with a language
    model and prompt template to form a complete RAG chain.

    Parameters
    -------------
    docs : list of Document
        List of LangChain Document objects used to initialize the BM25 retriever.
    vectorstore : object
        A vector store (e.g., FAISS, Chroma) that supports `.as_retriever()`
        for semantic search.
    generator : object
        A HuggingFace text-generation pipeline used as the language model.
    prompt : str
        A prompt template string containing placeholders (e.g., {context}, {query}).
    k : int, optional
        Number of top documents to retrieve for semantic search (default is 5).

    Returns
    -------------
    Runnable
        A LangChain Runnable pipeline that takes a query as input and returns
        a generated response.
    """
    # BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(
        docs,  
        k=5  
        )
    
    # Semantic retriever
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k} 
    )
    
    # Hybrid ensemble
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )
    
    llm = HuggingFacePipeline(pipeline=generator)
    prompt = ChatPromptTemplate.from_template(prompt)
    
    rag_chain = (
        {
            "context": ensemble_retriever | build_context,
            "query": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def hybrid_rag_pipeline(rag_chain, query):
    """
    Execute a query using a pre-initialized hybrid RAG pipeline.

    This function passes a user query through the RAG pipeline, which performs
    hybrid retrieval, constructs context, and generates a response using a
    language model.

    Parameters
    -------------
    rag_chain : Runnable
        The hybrid RAG pipeline created by `initialize_hybrid_rag`.
    query : str
        The user query string.

    Returns
    -------------
    str
        The generated response from the language model.
    """
    return print(rag_chain.invoke(query))