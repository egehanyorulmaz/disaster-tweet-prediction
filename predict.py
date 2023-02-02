import pandas as pd
from helpers.utils import load_model

# Load the models
logreg = load_model('models/logreg.pkl')
count_vectorizer1 = load_model('models/count_vectorizer1.pkl')
logreg2 = load_model('models/logreg2.pkl')
count_vectorizer3 = load_model('models/count_vectorizer3.pkl')
nb = load_model('models/nb.pkl')


## Prediction
df = pd.read_csv('data/test_preprocessed.csv')
df = df.drop(columns=['id', 'location'])
x_test = count_vectorizer1.transform(df.drop(columns=['keyword']))
x_test_2 = count_vectorizer3.transform(df.drop(columns=['keyword']))
x_test_3 = count_vectorizer3.transform(df.drop(columns=['text']))


# Ensemble
prediction1 = logreg.predict(x_test)
prediction2 = nb.predict(x_test_2)
prediction3 = logreg2.predict(x_test_3)

pred_final = (prediction1 + prediction2 + prediction3) / 3.0
pred_final = [1 if el>=0.5 else 0 for el in pred_final]

df = pd.DataFrame({'id': df['id'], 'target': pred_final})
df.to_csv('data/submission.csv', index=False)

a = 5
