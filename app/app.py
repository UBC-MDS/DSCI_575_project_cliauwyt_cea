from shiny import ui, render, App, reactive
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
from src.config import (
    BM25_INDEX_PATH,
    PREPROCESSED_CORPUS_PATH,
    SEMANTIC_INDEX_PATH,
    VECTOR_STORE_DIR,
)
from src.agent import build_agent, invoke_agent, format_tools, format_response

load_dotenv()

data = pd.read_csv(PREPROCESSED_CORPUS_PATH)

# Search
# BM25 and semantic search
# Semantic search embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# RAG
# Docs and vector store
docs = csv_loader(PREPROCESSED_CORPUS_PATH)
vectorstore = load_vectorstore(VECTOR_STORE_DIR)

# Retrievers
ensemble_retriever = hybrid_retriever(
    bm25_retriever(docs, k=3), 
    semantic_retriever(vectorstore, k=3)
)

# LLM
llm = load_llm()
agent = build_agent()


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
        ),

        ui.nav_panel(
            "Agent Mode",
            ui.layout_columns(
                ui.input_text("agent_query", "Query"),
                ui.div(
                    ui.output_ui("agent_text"),
                    # ui.output_ui("agent_tool")
                ),
                col_widths=(3, 9)
            )
        )
    )
)


def server(input, output, session):
    @render.data_frame
    def search_results():
        """Return search results based on the selected retrieval method."""
        type = input.retrieval_type()
        q = input.query()

        if type == "BM25":
            from src.bm25 import bm25_search

            results = (
                bm25_search(q, BM25_INDEX_PATH, data, top_k=3)
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
                semantic_search(q, SEMANTIC_INDEX_PATH, model, data, top_k=3)
                .assign(
                    review_text=lambda d: d["review_text"].str.slice(0, 200),
                    score=lambda d: d["score"].map(lambda x: f"{x:.2f}"),
                )
                .rename(columns=lambda c: c.replace("_", " ").title())
            )
            return render.DataTable(results)

    @render.ui
    def rag_text():
        """Render the RAG-generated answer text for the current query."""
        q = input.rag_query()
        rag_chain = initialize_rag_chain(ensemble_retriever, llm, prompt)
        return ui.markdown(rag_chain.invoke(q))
    
    @render.data_frame
    def rag_results():
        """Display retrieved documents used for RAG response generation."""
        q = input.rag_query()
        retreived_docs = ensemble_retriever.invoke(q)
        return render.DataTable(format_docs_to_df(retreived_docs))

    @reactive.calc
    def agent_response():
        """Compute and cache the agent response for the current query."""
        q = input.agent_query()
        return invoke_agent(agent, q)

    # @render.ui
    # def agent_tool():
    #     """Render tool-call output as markdown text."""
    #     return ui.markdown(format_tools(agent_response()))
    
    @render.ui
    def agent_text():
        """Render the final agent response as markdown text."""
        return ui.markdown(format_response(agent_response()))


app = App(app_ui, server)
