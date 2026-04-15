# DSCI_575_project_claudia-liauw_cynthiaagata

## Building a Smart Amazon Product Query Assistant

In this project, we built a context-aware product search assistant that returns relevant Amazon products from the Health and Personal Care category based on natural language queries. We compared keyword retrieval with BM25 and semantic retrieval with pre-trained embeddings and FAISS.

### Data

The data used is from [https://amazon-reviews-2023.github.io/](https://amazon-reviews-2023.github.io/). It consists of reviews data and metadata about the products. As the dataset is very large, we limited the number of rows to 20000.

### Preprocessing

We only retained records where the name of the product (`product_title`) is available. We build the corpus from the following fields from the metadata: `product_title`, `main_category`, `store`, and the `title` and `text` from reviews data.

For preprocessing, we removed stop words, tokens that are shorter than two characters, irrelevant parts of speech, no vector (out of vocabulary), and any non-alphabetic characters. We lemmatized the words and converted them to lower case.

### Retrieval Workflows

#### BM25

BM25 is an enhanced TF-IDF (term frequency-inverse document frequency) vectorisation. Each document (product comprising metadata and review data) is represented by a vector of the same length as the vocabulary (sparse) with TF-IDF scores for each word. This is saved as an index.

At retrieval time, the query is vectorised in the same way and BM25 calculates a score for each document. We retrieve the top k documents with the highest scores.

#### Semantic

Each document is represented as a dense vector of pre-trained embeddings. This is saved as an index with FAISS. 

At retrieval time, the query is vectorised in the same way and the FAISS index is used to perform approximate nearest neighbour search to retrieve the top k similar documents.

## Instructions

### Setup

**Prerequisites:** Python 3.9 or higher

1. Clone the repo
```bash
git clone https://github.com/UBC-MDS/DSCI_575_project_cliauwyt_cea
cd DSCI_575_project_cliauwyt_cea
```

2. Create and activate virtual environment
```bash
python -m venv env
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

4. Download the data: run the notebook `notebooks/download_data.ipynb`.
```bash
jupyter execute notebooks/download_data.ipynb
```

#### BM25 and semantic indices

Run the notebook `notebooks/milestone1_results.ipynb` to preprocess and save the data, and build and save BM25 and semantic indices:
```bash
jupyter execute notebooks/milestone1_results.ipynb
```

To preprocess and save the data only (overwrites existing file): 
```bash
python src/preprocess.py
```

#### RAG

To build and save vector store (overwrites existing files): 
```bash
python src/rag_pipeline.py
```

### Run the app

**Prerequisites:** BM25 and semantic indices

```bash
shiny run app/app.py
```