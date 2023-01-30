import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from prepare_data import Preprocessor
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from helpers.basic_models import fit_model


df = pd.read_csv('data/train_preprocessed.csv')

print('Splitting the data...')
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.2, random_state=42)
print('Data is successfully splitted.')

nb = MultinomialNB()
count_vec2 = CountVectorizer(stop_words='english', ngram_range=(1, 3))

print('Fitting the model...')
count_vectorizer1, nb = fit_model(nb, count_vec2, X_train, X_test, y_train, y_test, get_feature_importance=True)

logreg = LogisticRegression(max_iter=10000)
count_vectorizer2, logreg = fit_model(logreg, count_vec2, X_train, X_test, y_train, y_test, get_feature_importance=True)

df = pd.read_csv('data/train_preprocessed.csv')
x_test = count_vectorizer1.transform(df['text'])
predictions = logreg.predict(x_test)
df = pd.DataFrame({'id': df['id'], 'target': predictions})
df.to_csv('data/submission.csv', index=False)

a = 5





