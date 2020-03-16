# %%
import keras
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from nltk import PorterStemmer
from nltk.corpus import stopwords
import string
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
raw_obama = pd.read_pickle("data/tweets_obama.pickle")
df_obama = pd.DataFrame(
    {"text": raw_obama, "author": pd.Series(["Obama"]*len(raw_obama))})

raw_trump = pd.read_pickle("data/tweets_trump.pickle")
df_trump = pd.DataFrame(
    {"text": raw_trump, "author": pd.Series(["Trump"]*len(raw_trump))})

df = pd.concat([df_obama, df_trump]).reset_index().drop("index", axis=1)
df["author"] = df.author.astype("category")

# %%


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


# %%

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

# TT-Split
xtrain_raw, xtest_raw, ytrain, ytest = train_test_split(
    df.text, df.author, random_state=1234)

# Fit vectorizer on tweets
vectorizer = CountVectorizer()
vectorizer.fit(xtrain_raw)

# Transform tweets two vectors
_vocab_size = len(vectorizer.vocabulary_)
xtrain = vectorizer.transform(xtrain_raw)
xtest = vectorizer.transform(xtest_raw)

# Map author to (0,1)
author_map = {"Obama": 0, "Trump": 1}
ytrain = ytrain.map(author_map)
ytest = ytest.map(author_map)
# %%


def eval_model(model, is_keras=False):
    """ For a sklearn model, prints confusion matrix & classification report on test set. """
    print(f"====== Model: {model.__class__} ======")
    if is_keras:
        preds = model.predict_classes(xtest)
    else:
        preds = model.predict(xtest)
    print(confusion_matrix(ytest, preds))
    print()
    print(classification_report(ytest, preds))
    print("="*(12+2+7+len(str(model.__class__))))


# %%
# Baseline Model: Naive Bayes
naive_bayes = MultinomialNB()
naive_bayes.fit(xtrain, ytrain)
eval_model(naive_bayes)

# %%
# Second Basline Model: Logistic Regression
logistic_regression = LogisticRegression()
logistic_regression.fit(xtrain, ytrain)
eval_model(logistic_regression)

# %%
# Neural Network
neural_net = keras.models.Sequential([
    keras.layers.Dense(units=10, input_shape=(_vocab_size,), activation="sigmoid"),
    keras.layers.Dense(units=1, activation="sigmoid")
])

neural_net.compile(optimizer="adam", loss="binary_crossentropy")
N_EPOCHS = 30
neural_net.fit(x=xtrain, y=ytrain, epochs=N_EPOCHS, batch_size=64, validation_data=(xtest, ytest))

# %%
sns.lineplot(x=np.arange(N_EPOCHS), y=neural_net.history.history["val_loss"], label="VALIDATION")
sns.lineplot(x=np.arange(N_EPOCHS), y=neural_net.history.history["loss"], label="TRAINING")
eval_model(neural_net, is_keras=True)

# %%
results = pd.DataFrame({"text": xtest_raw, "true": ytest, "pred_nn": neural_net.predict_classes(xtest).reshape(1, -1)[0], "pred_lr": logistic_regression.predict(xtest)})

# %%
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# For embedding: Assign a number to each word (e.g.: obama -> 1) instead of one hot encoding/count vectorizing
tk = Tokenizer(num_words=5000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n',lower=True, split=" ")
tk.fit_on_texts(xtrain_raw)
xtrain_emb = tk.texts_to_sequences(xtrain_raw)
xtrain_emb = pad_sequences(xtrain_emb, 280)
xtest_emb = tk.texts_to_sequences(xtest_raw)
xtest_emb = pad_sequences(xtest_emb, 280)

# Embedding Modle
emb_input = keras.layers.Input(shape=(280,))
embedding_layer = keras.layers.Embedding(input_dim=_vocab_size, output_dim=15, input_length=280)(emb_input)
flattened_embedding_layer = keras.layers.Flatten()(embedding_layer)
final_layer = keras.layers.Dense(units=1, activation="sigmoid")(flattened_embedding_layer)
emb_model = keras.Model(inputs=emb_input, outputs=final_layer)

# Compile and fit
emb_model.compile(optimizer="adam", loss="binary_crossentropy")
N_EPOCHS = 50
emb_model.fit(x=xtrain_emb, y=ytrain, epochs=N_EPOCHS, batch_size=64, validation_data=(xtest_emb, ytest)) #validation_data=([xtest], [ytest]))

# %%
sns.lineplot(x=np.arange(N_EPOCHS), y=emb_model.history.history["val_loss"], label="VALIDATION")
sns.lineplot(x=np.arange(N_EPOCHS), y=emb_model.history.history["loss"], label="TRAINING")
eval_model(neural_net, is_keras=True)

# %%
# two sample embeddings, 1 word = 15 dimensions
emb_model.layers[1].get_weights()[0].shape
w1_emb = emb_model.layers[1].get_weights()[0][tk.texts_to_sequences(["bad this is gonna be huge"])]
w2_emb = emb_model.layers[1].get_weights()[0][tk.texts_to_sequences(["good"])]

print(w1_emb)
print(w2_emb)

print()
print("====== Model: 15-Dimensional Word Embedding ======")
preds = pd.Series((emb_model.predict(pad_sequences(tk.texts_to_sequences(xtest_raw), 280)) > 0.5).reshape(1,-1)[0]).map({True:1, False:0})
print(confusion_matrix(ytest, preds))
print()
print(classification_report(ytest, preds))
print("="*50)