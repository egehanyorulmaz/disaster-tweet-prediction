from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

import pandas as pd


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
