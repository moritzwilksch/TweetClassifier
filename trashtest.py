# %%
import pandas as pd
# sys.path.append("..") to import from parent package
import createTwitterAPI
import tweepy

api = createTwitterAPI.create()


def get_tweets(username: str, fetch_from_disk: bool = True) -> pd.DataFrame:
    if not fetch_from_disk:
        raw = [tweet for tweet in tweepy.Cursor(api.user_timeline, id=username,
                                                tweet_mode="extended", include_rts=False,
                                                wait_on_rate_limit=True, since_id='2019-10-01').items(10000)]

        id = [tweet.id for tweet in raw]
        date = [tweet.created_at for tweet in raw]
        text = [tweet.full_text for tweet in raw]

        return pd.DataFrame({'id': id, 'date': date, 'text': text})


# %%
# tweets = get_tweets("sixtus", fetch_from_disk=False)

# %%
df = pd.read_pickle('/Users/Moritz/Desktop/tweets2.pickle')

# %%
df['hit'] = df['text'].str.lower().str.contains('arschloch')


#%%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = sns.load_dataset('tips')

"""group_a = df.query("day=='Thur'")['tip']
group_b = df.query("day=='Sun'")['tip']
"""

group_a = df.query("time=='Lunch'")['tip']
group_b = df.query("time=='Dinner'")['tip']

sns.distplot(group_a)
sns.distplot(group_b)
plt.title("Distribution of Tips.")
plt.show()

# %%

# Number of Bootstraps doesn't really affect Credible Interval, ONLY ITS CONVERGENCE!
n_bootstraps = 1000

# Each sample drawn should be as big as the original sample (len(group_a))!!!!
# Drawing larger samples would narrow the resulting Credible Interval  thereby simulating
# what CI would result if the ORIGINAL SAMPLE HAD BEEN BIGGER TO BEGIN WITH!!!!!
# This can be used for minimum sample size estimation.
mean_a = np.mean(np.random.choice(group_a, (n_bootstraps, len(group_a))), axis=1)
mean_b = np.mean(np.random.choice(group_b, (n_bootstraps, len(group_b))), axis=1)
# Bootstraped sample size HEAVILY AFFECTS CI (variance of the mean of 10M values sampled from 68
# is waaaaaaay lower than variance of the mean of 68 samples sampled from 68


#%%
sns.distplot(mean_a)
sns.distplot(mean_b)
plt.title(f"Mean Tip. {n_bootstraps} Bootstraps.")
plt.show()

#%%
diff_means = mean_b - mean_a
lower, upper = np.round(np.percentile(diff_means, (2.5, 97.5)), 4)
sns.distplot(diff_means, kde=False)
plt.title(fr"Difference in Means. 95CI: {(lower, upper)}")
plt.axvline(lower, color='0.5')
plt.axvline(upper, color='0.5')
plt.show()

#%%
l = np.array([1]*6+[0]*95)
