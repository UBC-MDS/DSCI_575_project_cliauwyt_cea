from shiny import ui, render, App
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vectorstore import csv_loader, load_vectorstore
from src.rag_pipeline import load_llm, semantic_retriever, initialize_rag_chain, format_docs_to_df
from src.hybrid import bm25_retriever, hybrid_retriever
from src.prompts import prompt

load_dotenv()

corpus_path = 'data/processed/preprocessed_corpus.csv'
data = pd.read_csv(corpus_path)

# Search
# BM25 and semantic search
bm25_index_path = 'data/processed/bm25.pkl'
semantic_index_path = 'data/processed/embedding.faiss'

# Semantic search embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# RAG
# Docs and vector store
docs = csv_loader(corpus_path)
vector_path = "data/processed/vector_store"
vectorstore = load_vectorstore(vector_path)

# Retrievers
ensemble_retriever = hybrid_retriever(
    bm25_retriever(docs, k=3), 
    semantic_retriever(vectorstore, k=3)
)

# LLM
llm = load_llm()


app_ui = ui.page_fillable(
    ui.panel_title("Health and Personal Care Search"),
    ui.tags.style("""
        .shiny-input-radiogroup > .control-label {
            margin-bottom: 0.9rem;
        }

        #search_results table td:nth-child(1),
        #search_results table th:nth-child(1) {
            min-width: 320px;
            max-width: 320px;
            white-space: normal;
        }

        #search_results table td:nth-child(2),
        #search_results table th:nth-child(2) {
            min-width: 460px;
            max-width: 460px;
            white-space: normal;
        }

        #rag_results table td:nth-child(2),
        #rag_results table th:nth-child(2) {
            min-width: 320px;
            max-width: 320px;
            white-space: normal;
        }

        #rag_results table td:nth-child(4),
        #rag_results table th:nth-child(4) {
            min-width: 400px;
            max-width: 400px;
            white-space: normal;
        }

        #rag_text {
            display: block;
            margin-bottom: 1rem;
        }
    """),
    ui.navset_pill(  
        ui.nav_panel(
            "Search Only",
            ui.layout_columns(
                ui.div(
                    ui.input_radio_buttons(
                        "retrieval_type",
                        "Retreival type",
                        ["BM25", "Semantic"]
                    ),
                    ui.input_text("query", "Query")
                ),
                ui.output_data_frame("search_results"),
                col_widths=(3, 9)
            )
        ),
        ui.nav_panel(
            "RAG Mode",
            ui.layout_columns(
                ui.input_text("rag_query", "Query"),
                ui.div(
                    ui.output_ui("rag_text"),
                    ui.output_data_frame("rag_results")
                ),
                col_widths=(3, 9)
            )
        )
    )
)


def server(input, output, session):
    @render.data_frame
    def search_results():
        type = input.retrieval_type()
        q = input.query()

        if type == "BM25":
            from src.bm25 import bm25_search

            results = (
                bm25_search(q, bm25_index_path, data, top_k=3)
                .assign(
                    review_text=lambda d: d["review_text"].str.slice(0, 200),
                    score=lambda d: d["score"].map(lambda x: f"{x:.2f}"),
                )
                .rename(columns=lambda c: c.replace("_", " ").title())
            )
            return render.DataTable(results)

        elif type == "Semantic":
            from src.semantic import semantic_search

            results = (
                semantic_search(q, semantic_index_path, model, data, top_k=3)
                .assign(
                    review_text=lambda d: d["review_text"].str.slice(0, 200),
                    score=lambda d: d["score"].map(lambda x: f"{x:.2f}"),
                )
                .rename(columns=lambda c: c.replace("_", " ").title())
            )
            return render.DataTable(results)

    @render.ui
    def rag_text():
        q = input.rag_query()
        rag_chain = initialize_rag_chain(ensemble_retriever, llm, prompt)
        return ui.markdown(rag_chain.invoke(q))
    
    @render.data_frame
    def rag_results():
        q = input.rag_query()
        retreived_docs = ensemble_retriever.invoke(q)
        return render.DataTable(format_docs_to_df(retreived_docs))


app = App(app_ui, server)
