import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from helpers.basic_models import fit_model
from helpers.utils import save_model

df = pd.read_csv('data/train_preprocessed.csv')

print('Splitting the data...')
y = df['target']
X = df.drop(columns=['id', 'location', 'target'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Data is successfully splitted.')

print('Fitting the NB model...')
nb = MultinomialNB()
count_vec2 = CountVectorizer(stop_words='english', ngram_range=(1, 3))
count_vectorizer1, nb = fit_model(nb, count_vec2,
                                  X_train.drop(columns=['keyword']),
                                  X_test.drop(columns=['keyword']), y_train, y_test,
                                  column='text',
                                  get_feature_importance=False)

print('Fitting the LogisticRegression model...')
logreg = LogisticRegression(max_iter=10000)
count_vectorizer2, logreg = fit_model(logreg, count_vec2,
                                      X_train.drop(columns=['keyword']),
                                      X_test.drop(columns=['keyword']), y_train, y_test,
                                      column='text',
                                      get_feature_importance=False)

print('Fitting the LogisticRegression with Keyword column')
logreg2 = LogisticRegression(max_iter=10000)
count_vec3 = CountVectorizer()
count_vectorizer3, logreg2 = fit_model(logreg2, count_vec3,
                                       X_train.drop(columns=['text']),
                                       X_test.drop(columns=['text']), y_train, y_test,
                                       column='keyword',
                                       get_feature_importance=True)

# save the model
save_model(count_vectorizer1, '/models/count_vectorizer1.pkl')
save_model(count_vectorizer2, '/models/count_vectorizer2.pkl')
save_model(count_vectorizer3, '/models/count_vectorizer3.pkl')
save_model(logreg, '/models/logreg.pkl')
save_model(logreg2, '/models/logreg2.pkl')
save_model(nb, '/models/nb.pkl')
print('Models are successfully saved.')