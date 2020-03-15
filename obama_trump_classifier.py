# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sys.path.append("..") to import from parent package
import createTwitterAPI
import tweepy

# %%
def get_tweets(username: str, fetch_from_disk: bool = True) -> pd.DataFrame:
    if not fetch_from_disk:
        api = createTwitterAPI.create()
        status = [tweet._json["full_text"] for tweet in tweepy.Cursor(api.user_timeline, id=username, tweet_mode="extended", include_rts=False).items(3000)]        
        return status

# %%
tweets_obama = pd.Series(get_tweets("BarackObama", fetch_from_disk=False))
# tweets_trump = pd.Series(get_tweets("realDonaldTrump", fetch_from_disk=False))