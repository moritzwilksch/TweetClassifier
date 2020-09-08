# %%
import keras
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessingPL import create_pipeline
import logging
import time
# %%
raw_obama = pd.read_pickle("data/tweets_obama.pickle")
df_obama = pd.DataFrame(
    {"text": raw_obama, "author": pd.Series(["Obama"] * len(raw_obama))})

raw_trump = pd.read_pickle("data/tweets_trump.pickle")
df_trump = pd.DataFrame(
    {"text": raw_trump, "author": pd.Series(["Trump"] * len(raw_trump))})

df = pd.concat([df_obama, df_trump]).reset_index().drop("index", axis=1)
df["author"] = df.author.astype("category")

# %%
cleaning_pipeline = create_pipeline()

# %%
df = cleaning_pipeline.fit_transform(df)

# %%

# TT-Split
xtrain_raw, xtest_raw, ytrain, ytest = train_test_split(
    df.text, df.author, random_state=1234)

author_map = {"Obama": 0, "Trump": 1}
ytrain = ytrain.map(author_map)
ytest = ytest.map(author_map)

# %%
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# fit on train
tok = Tokenizer()
tok.fit_on_texts(xtrain_raw.values)

# transform train
xtrain_seq = tok.texts_to_sequences(xtrain_raw)
xtrain_seq = pad_sequences(xtrain_seq, maxlen=280)

# transform test
xtest_seq = tok.texts_to_sequences(xtest_raw)
xtest_seq = pad_sequences(xtest_seq, maxlen=280)

# %%
n_dim = 10

model = keras.models.Sequential([
    keras.layers.Embedding(name="Embedding", input_dim=max(tok.index_word) + 1, output_dim=n_dim, input_length=280),
    # keras.layers.Reshape((n_dim,)),
    keras.layers.Flatten(),
    # keras.layers.Dense(name="Dense", units=10, activation='sigmoid'),
    keras.layers.Dense(name="DenseOutput", units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(
    x=xtrain_seq,
    y=ytrain,
    epochs=15,
    validation_data=(xtest_seq, ytest),
    batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(patience=3)],
    # class_weight={0: 1.0, 1: 2.0}

)

pd.DataFrame({'train': model.history.history['loss'], 'test': model.history.history['val_loss']}).plot()
plt.show()

model.save('embeddingModel.hd5')

def eval_model(model, is_keras=False):
    """ For a sklearn model, prints confusion matrix & classification report on test set. """
    print(f"====== Model: {model.__class__} ======")
    if is_keras:
        preds = model.predict_classes(xtest_seq)
    else:
        preds = model.predict(xtest_seq)
    print(confusion_matrix(ytest, preds))
    print()
    print(classification_report(ytest, preds))
    print("=" * (12 + 2 + 7 + len(str(model.__class__))))


eval_model(model, is_keras=True)

logging.basicConfig(filename='modellog.log', level=logging.DEBUG)
logging.warning("#" * 20 + time.asctime() + "#"*20)
logging.info(model.summary(print_fn=lambda x: logging.info(x)))
logging.info(classification_report(ytest, model.predict_classes(xtest_seq)))


#%%
from scipy.spatial.distance import cosine
a = model.layers[0].get_weights()[0][tok.texts_to_sequences(["health"])]
b = model.layers[0].get_weights()[0][tok.texts_to_sequences(["trump"])]
1-cosine(a,b)

#%%
n_dim = 10

modelcnn = keras.models.Sequential([
    keras.layers.Embedding(name="Embedding", input_dim=max(tok.index_word) + 1, output_dim=n_dim, input_length=280),
    keras.layers.Conv1D(filters=16, kernel_size=3),
    keras.layers.MaxPool1D(3),
    keras.layers.Conv1D(filters=16, kernel_size=3),
    keras.layers.Flatten(),
    keras.layers.Dense(name="DenseOutput", units=1, activation='sigmoid')
])

modelcnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
"""modelcnn.fit(
    x=xtrain_seq,
    y=ytrain,
    epochs=15,
    validation_data=(xtest_seq, ytest),
    batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(patience=3)],
    # class_weight={0: 1.0, 1: 2.0}

)"""

pd.DataFrame({'train': modelcnn.history.history['loss'], 'test': modelcnn.history.history['val_loss']}).plot()
plt.show()

model.save('embeddingModelCNN.hd5')