import pandas as pd
import spacy
import os


def make_corpus(df: pd.DataFrame, cols: list, asin: str, meta: list = []) -> pd.DataFrame:
    """
    Make a corpus DataFrame that combines all given text columns
    into one text column and extract the asin column. The corpus will
    be for information retrieval.

    Parameters
    -------------
    df : pd.DataFrame
        clean datadrame
    cols : list
        list of text columns to combine into one text in corpus
    asin : str
        the identification column in clean data

    Returns
    -------------
    pd.Dataframe: corpus result
    """
    corpus = pd.DataFrame({})
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


def make_corpus_preprocess(raw_data_path, corpus_path):
    # Read data and drop missing values
    import duckdb
    c2 = duckdb.connect()
    data = c2.execute(f"SELECT * FROM read_parquet('{raw_data_path}')").df()
    data.dropna(subset=['product_title'], inplace=True)

    # Extract fields for retrieval
    cols = ['product_title', 'main_category', 'store', 'title', 'text']
    meta = ['product_title', 'rating', 'text']

    corpus = make_corpus(df=data, cols=cols, asin="asin", meta=meta)

    # preprocess corpus and save it
    os.makedirs('data/processed', exist_ok=True)
    nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])
    corpus["text"] = [preprocess_spacy(text) for text in nlp.pipe(corpus["text"])]
    corpus.to_csv(corpus_path)


if __name__ == "__main__":
    raw_data_path = 'data/raw/merged.parquet'
    corpus_path = 'data/processed/preprocessed_corpus.csv'
    make_corpus_preprocess(raw_data_path, corpus_path)
    print(f"Saved corpus to {corpus_path}")
