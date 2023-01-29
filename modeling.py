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


def fit_model(model, count_vectorizer, X_train, X_test, y_train, y_test, get_feature_importance=True):
    print("Model inputs are: ")
    print(count_vectorizer)
    print(model)
    print("Fitting the Count Vectorizer")
    count_vectorizer.fit(X_train)

    X_train_dtm = count_vectorizer.fit_transform(X_train)
    X_test_dtm = count_vectorizer.transform(X_test)
    print("CountVectorizer is successfuly fitted for train and test data!")

    model.fit(X_train_dtm, y_train)
    y_pred_class = model.predict(X_test_dtm)
    print(f"Test Accuracy: {metrics.accuracy_score(y_test, y_pred_class) * 100:.1f}%")

    print(classification_report(y_test, y_pred_class))

    if get_feature_importance:
        print("The most important features")
        print("-" * 20)
        feature_names = count_vectorizer.get_feature_names_out()
        if type(model) == LogisticRegression:
            feature_importance = model.coef_[0]
        elif type(model) == MultinomialNB:
            # if model is Naive Bayes then we can get the feature importance
            feature_importance = model.feature_log_prob_[1, :].argsort()[::-1]
        coefs_with_fns = zip(feature_names, feature_importance)

        coefs_with_fns_df = pd.DataFrame(coefs_with_fns,
                                         columns=['feature', 'coefficient'])

        coefs_with_fns_df.sort_values(by='coefficient', ascending=True, inplace=True)
        print(coefs_with_fns_df)

        print("The least important features")
        print("-" * 20)
        coefs_with_fns_df.sort_values(by='coefficient', ascending=False, inplace=True)
        print(coefs_with_fns_df)

    return count_vectorizer, model

if __name__ == '__main__':
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





