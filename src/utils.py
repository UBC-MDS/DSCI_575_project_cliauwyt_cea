import pandas as pd

def make_corpus(df: pd.DataFrame, cols: list, asin: str = None) -> pd.DataFrame:
    """
    Make a corpus DataFrame that combines all given text columns
    into one text column and extract the asin column. The corpus will
    be for information retrieval.
    """
    corpus = pd.DataFrame({})
    corpus['asin'] = df[asin]
    corpus['text']  = df[cols].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    return corpus
