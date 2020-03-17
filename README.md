# TweetClassifier
## Obama VS Trump Tweet Classification
Classifies short texts (tweets). Given a short text, this application predicts whether Obama or Trump is most likely the author.
Runs as flask app: Offers simple one-page input field with submit button to make prediction off of a short text.

## Technicals
The underlying model is a shallow neural net with a vocabulary size of 6471. There is only one densly connected layer with 10 neurons.


## Performance
====== Model: <class 'keras.engine.sequential.Sequential'> ======
### Confusion Matrix (Holdout Dataset)
#### 0 = Obama, 1 = Trump

|  |0  | 1 |
|--|---|---|
|0 |718| 12|
|1 |16 | 316|

|              |precision|    recall|  f1-score |  support|
|--------------|---------|----------|-----------|---------|
|           0  |   0.98  |    0.98  |   0.98    |   730   |
|           1  |   0.96  |    0.95  |   0.96    |   332   |
|    accuracy  |         |          |   0.97    |  1062   |
|   macro avg  |   0.97  |    0.97  |   0.97    |  1062   |
|weighted avg  |   0.97  |    0.97  |   0.97    |  1062   |

=================================================================