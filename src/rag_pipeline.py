from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def semantic_retriever(vectorstore, k=5):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}  # Fetch k most similar documents
    )
    return retriever


def build_context(docs):
    return "\n\n".join(
        f"Product ASIN: {doc.metadata.get('asin', 'N/A')}\n"
        f"Title: {doc.metadata.get('product_title', 'N/A')}\n"
        f"Rating: {doc.metadata.get('rating', 'N/A')}/5.0\n"
        f"Review: {doc.metadata.get('review_text', 'N/A')}\n"
        for doc in docs
    )


def initialize_rag_chain(retriever, llm, prompt):
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
    return rag_chain.invoke(query)
