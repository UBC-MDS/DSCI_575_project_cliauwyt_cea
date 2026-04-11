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
    ui.input_radio_buttons(
        "retrieval_type",
        "Retreival type",
        ["BM25", "Semantic"]
    ),
    ui.input_text("query", "Query"),
    ui.output_data_frame("search_results")
)


def server(input, output, session):
    @render.data_frame
    def search_results():
        type = input.retrieval_type()
        q = input.query()

        if type == "BM25":
            pass

        elif type == "Semantic":
            from src.semantic import semantic_search

            model = SentenceTransformer("all-MiniLM-L6-v2")
            semantic_index_path = 'data/processed/embedding.faiss'
            results = semantic_search(q, semantic_index_path, model, data, top_k=3)
            return render.DataTable(results)


app = App(app_ui, server)
