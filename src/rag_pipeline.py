from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


def load_llm():
    """Initialize and return the configured Hugging Face chat LLM.

    Returns
    -------
    ChatHuggingFace
        Chat model wrapper backed by a Hugging Face inference endpoint.
    """
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        provider="novita"
    )

    return ChatHuggingFace(llm=llm_endpoint)


def semantic_retriever(vectorstore, k=5):
    """Create a semantic similarity retriever from a vector store.

    Parameters
    ----------
    vectorstore : object
        Vector store object.
    k : int, default=5
        Number of most similar documents to return.

    Returns
    -------
    object
        Retriever configured for similarity search.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}  # Fetch k most similar documents
    )
    return retriever


def build_context(docs):
    """Format retrieved documents into a prompt-ready context string.

    Parameters
    ----------
    docs : iterable
        Documents with metadata fields such as ASIN, product title, rating,
        and review text.

    Returns
    -------
    str
        Single string containing formatted document snippets separated by
        blank lines.
    """
    return "\n\n".join(
        f"Product ASIN: {doc.metadata.get('asin', 'N/A')}\n"
        f"Title: {doc.metadata.get('product_title', 'N/A')}\n"
        f"Rating: {doc.metadata.get('rating', 'N/A')}/5.0\n"
        f"Review: {doc.metadata.get('review_text', 'N/A')}\n"
        for doc in docs
    )


def initialize_rag_chain(retriever, llm, prompt):
    """Build a RAG chain that retrieves context and generates an answer.

    Parameters
    ----------
    retriever : object
        Retriever used to fetch relevant documents from the index.
    llm : object
        Language model runnable used to generate the final response.
    prompt : str
        Prompt template string expecting ``context`` and ``query``.

    Returns
    -------
    object
        Runnable chain that takes a query string and returns model output.
    """
    prompt = ChatPromptTemplate.from_template(prompt)

    rag_chain = (
        {
            "context": retriever | build_context,
            "query": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def invoke_rag_chain(rag_chain, query):
    """Invoke a prepared RAG chain with a user query.

    Parameters
    ----------
    rag_chain : object
        Runnable RAG chain returned by ``initialize_rag_chain``.
    query : str
        User question string.

    Returns
    -------
    str
        Chain output as a string.
    """
    return rag_chain.invoke(query)
