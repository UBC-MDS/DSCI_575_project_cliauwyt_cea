import pandas as pd
import spacy
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLEAN_DATA_PATH,
    PREPROCESSED_CORPUS_PATH,
    PROCESSED_DATA_DIR,
    RAW_MERGED_PARQUET_PATH,
)


def make_corpus(
        df: pd.DataFrame, 
        cols: list = ['product_title', 'main_category', 'store', 'title', 'text'], 
        asin: str = 'asin', 
        meta: list = ['product_title', 'rating', 'text'],
        ) -> pd.DataFrame:
    """
    Make a corpus DataFrame that combines all given text columns
    into one text column and extract the asin column. The corpus will
    be used for information retrieval and include meta columns for 
    including more information.

    Parameters
    -------------
    df : pd.DataFrame
        clean datadrame
    cols : list, default includes all text columns
        list of text columns to combine into one text in corpus
    asin : str, default is "asin"
        the identification column in clean data
    meta : str, default include product title, rating, and review text columns
        the meta column in clean data that we want to include in corpus

    Returns
    -------------
    pd.Dataframe: corpus result
    """
    corpus = pd.DataFrame()
    corpus['asin'] = df[asin]
    corpus['text']  = df[cols].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    for x in meta:
        if x == 'text':
            corpus['review_text'] = df[x]
        else:
            corpus[x] = df[x]
    return corpus

# Code adapted from DSCI 563 lab 3
def preprocess_spacy(
    doc,
    min_token_len=2,
    irrelevant_pos=["ADV", "PRON", "CCONJ", "PUNCT", "PART", "DET", "ADP", "SYM", "NUM"]):
    """
    Given text, min_token_len, and irrelevant_pos carry out preprocessing of the text
    and return a preprocessed string.

    Parameters
    -------------
    doc : spaCy doc object
        the spacy doc object of the text
    min_token_len : int
        min_token_length required
    irrelevant_pos : list
        a list of irrelevant pos tags

    Returns
    -------------
    str: the preprocessed text
    """

    clean_text = []

    for token in doc:
        if (
            token.is_stop == False  # Check if it's not a stopword
            and len(token) > min_token_len  # Check if the word meets minimum threshold
            and token.pos_ not in irrelevant_pos
            and not token.is_oov
            and token.is_alpha
        ):  # Check if the POS is in the acceptable POS tags
            lemma = token.lemma_  # Take the lemma of the word
            clean_text.append(lemma.lower())

    return " ".join(clean_text)

def make_clean_data(raw_data_path, clean_data_path):
    """
    Build and preprocess a clean data set from a raw parquet dataset.

    This function loads the raw merged review-product data and removes rows with
    missing product titles.

    Parameters
    -------------
    raw_data_path : str
        Path to the input parquet file.
    clean_data_path : str
        Destination path for the preprocessed data CSV.

    Returns
    -------------
    None
        The function saves the clean data to ``clean_data_path``.
    """
    import duckdb

    c2 = duckdb.connect()
    data = c2.execute(f"SELECT * FROM read_parquet('{raw_data_path}')").df()
    data.dropna(subset=['product_title'], inplace=True)

    # save clean data
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    data.to_csv(clean_data_path)
    print(f"Saved clean data to {clean_data_path}")


def make_corpus_preprocess(clean_data_path, corpus_path):
    """
    Build and preprocess the retrieval corpus from a clean dataset.

    This function loads the clean data, combines selected fields into a 
    single retrieval text per product/review, applies spaCy-based 
    token filtering and lemmatizing, and writes the final corpus to 
    a CSV file.

    Parameters
    -------------
    clean_data_path : str
        Path to the clean data file.
    corpus_path : str
        Destination path for the preprocessed corpus CSV.

    Returns
    -------------
    None
        The function saves the preprocessed corpus to ``corpus_path``.
    """
    data = pd.read_csv(clean_data_path)

    corpus = make_corpus(df=data)

    # preprocess corpus and save it
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])
    corpus["text"] = [preprocess_spacy(text) for text in nlp.pipe(corpus["text"])]
    corpus.to_csv(corpus_path)
    print(f"Saved preprocessed corpus to {corpus_path}")


if __name__ == "__main__":
    make_clean_data(RAW_MERGED_PARQUET_PATH, CLEAN_DATA_PATH)
    make_corpus_preprocess(CLEAN_DATA_PATH, PREPROCESSED_CORPUS_PATH)