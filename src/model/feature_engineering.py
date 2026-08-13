# modelo de baseline TF_IDF
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def transform_term_frequency(df: pd.DataFrame) -> pd.DataFrame:
    tv = TfidfVectorizer(stop_words='english',
                        ngram_range=(1, 2), min_df=0.2, max_df=0.8)
    tfidf = tv.fit_transform(df.medical_abstract)
    tfidf_df = pd.DataFrame(tfidf.toarray(), columns=tv.get_feature_names_out())

    return tfidf_df
