from shiny import ui, render, App
import duckdb
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

c2 = duckdb.connect()
data = c2.execute(
    "SELECT * FROM read_parquet('data/raw/merged.parquet')"
).df()
data.dropna(subset=['product_title'], inplace=True)

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
    """),
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
)


def server(input, output, session):
    @render.data_frame
    def search_results():
        type = input.retrieval_type()
        q = input.query()

        if type == "BM25":
            from src.bm25 import bm25_search

            bm25_index_path = 'data/processed/bm25.pkl'
            results = (
                bm25_search(q, bm25_index_path, data, top_k=3)
                .assign(
                    text=lambda d: d["text"].str.slice(0, 200),
                    score=lambda d: d["score"].map(lambda x: f"{x:.2f}"),
                )
                .rename(columns=lambda c: c.replace("_", " ").title())
            )
            return render.DataTable(results)

        elif type == "Semantic":
            from src.semantic import semantic_search

            model = SentenceTransformer("all-MiniLM-L6-v2")
            semantic_index_path = 'data/processed/embedding.faiss'
            results = (
                semantic_search(q, semantic_index_path, model, data, top_k=3)
                .assign(
                    text=lambda d: d["text"].str.slice(0, 200),
                    score=lambda d: d["score"].map(lambda x: f"{x:.2f}"),
                )
                .rename(columns=lambda c: c.replace("_", " ").title())
            )
            return render.DataTable(results)


app = App(app_ui, server)
