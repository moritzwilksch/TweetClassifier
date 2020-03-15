# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
raw_obama = pd.read_pickle("tweets_obama.pickle")
df_obama = pd.DataFrame({"text": raw_obama, "author": pd.Series(["Obama"]*len(raw_obama))})

raw_trump = pd.read_pickle("tweets_trump.pickle")
df_trump = pd.DataFrame({"text": raw_trump, "author": pd.Series(["Trump"]*len(raw_trump))})

df = pd.concat([df_obama, df_trump]).reset_index().drop("index", axis=1)
df["author"] = df.author.astype("category")

# %%
import re
import string
from nltk.corpus import stopwords
from nltk import PorterStemmer

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
    df["text"] = df.text.apply(lambda s: "".join([char for char in s if char not in string.punctuation]).lower())
    return df

def remove_stopwords(df):
    stopwords_en = stopwords.words("english")
    df["text"] = df.text.apply(lambda s: " ".join([word for word in s.split() if word not in stopwords_en]))
    return df

def stem_words(df):
    stemmer = PorterStemmer()
    df["text"] = df.text.apply(lambda s: stemmer.stem(s))
    return df

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

cleaning_pipeline = Pipeline([
    ("Start", FunctionTransformer(start_pipeline)),
    ("Remove links", FunctionTransformer(remove_link)),
    ("Remove mentions", FunctionTransformer(remove_mentions)),
    ("Remove punctuation", FunctionTransformer(remove_punctuation)),
    ("Stem words", FunctionTransformer(stem_words)),
    ("Remove stopwords", FunctionTransformer(remove_stopwords)),
])

# %%
df = cleaning_pipeline.fit_transform(df)

# %%
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(df.text, df.author, random_state=1234)

# %%
