# %%
# FLASK Imports
from flask import *
import sys
sys.path.append("/Users/Moritz/Documents/DATEN/Coding/DataScience/TweetClassifier/")

# Standard Imports
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import keras
import numpy as np
import pandas as pd
import pickle

# Custom Imports
from preprocessingPL import create_pipeline


app = Flask(__name__)



def predict_from_string(s:str) -> int:
    """ Given a string of text (tweet), predicts whether the autor is Obama (0) or Trump (1) """
    df = pd.DataFrame({"text":[s]})
    df = pl.transform(df)
    vec = vectorizer.transform(df["text"])
    pred = model.predict_classes(vec)[0,0]
    return pred

@app.route("/")
def hw():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    s = request.form["text"]
    pred = "Obama" if predict_from_string(str(s)) == 0 else "Trump"
    return render_template("index.html", prediction=pred)

if __name__ == "TweetClassifier.app":
    app.debug = True
    # Loading Model and Pipeline and Vectorizer
    pl = create_pipeline()
    vectorizer = pickle.load(open("models/vectorizer.pickle", "rb"))
    model = keras.models.load_model("models/classifierNeuralNet.hd5")
    app.run()
