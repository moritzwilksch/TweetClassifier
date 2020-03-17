from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from nltk import PorterStemmer
from nltk.corpus import stopwords
import string
import re
import pandas as pd

def create_pipeline():
    def start_pipeline(df):
        """ Creates copy of df to prevent side effects. """
        return df.copy()


    def remove_link(df):
        link = re.compile(" http.*$")
        df["text"] = df.text.apply(lambda s: re.sub(link, "", s))
        return df


    def remove_mentions(df):
        mention = re.compile("@\w*")
        df["text"] = df.text.apply(lambda s: re.sub(mention, "", s))
        return df


    def remove_punctuation(df):
        df["text"] = df.text.apply(lambda s: "".join(
            [char for char in s if char not in string.punctuation]).lower())
        return df


    def remove_stopwords(df):
        stopwords_en = stopwords.words("english")
        df["text"] = df.text.apply(lambda s: " ".join(
            [word for word in s.split() if word not in stopwords_en]))
        return df


    def stem_words(df):
        stemmer = PorterStemmer()
        df["text"] = df.text.apply(lambda s: stemmer.stem(s))
        return df

    cleaning_pipeline = Pipeline([
        ("Start", FunctionTransformer(start_pipeline)),
        ("Remove links", FunctionTransformer(remove_link)),
        ("Remove mentions", FunctionTransformer(remove_mentions)),
        ("Remove punctuation", FunctionTransformer(remove_punctuation)),
        ("Stem words", FunctionTransformer(stem_words)),
        ("Remove stopwords", FunctionTransformer(remove_stopwords)),
    ])

    return cleaning_pipeline