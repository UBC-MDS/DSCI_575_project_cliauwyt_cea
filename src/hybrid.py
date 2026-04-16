import spacy 
from src.bm25 import preprocess_spacy
from langchain_core.runnables import RunnablePassthrough



def preprocess_query(query):
    nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])
    query = preprocess_spacy(nlp(query))
    return query


def hybrid_pipeline(hybrid_retriever, generator, query, prompt, k=5):
    rag_chain = (
        {
            "context": hybrid_retriever | build_context,
            "question": RunnablePassthrough()
        }
    )