## Natural Language Processing to Predict Disaster Tweets
This repository is focusing on solving the Kaggle Competition, ['Natural Language Processing with Disaster Tweets'](https://www.kaggle.com/competitions/nlp-getting-started/overview). Although the competition is considered as an easy start for the beginners to NLP, I will do my best to develop codes in a more clean and structured way instead of following a quick/messy approach and achieve 100% score. If you see a mistake or areas of improvement, I am open to suggestions and I would be happy to talk.


## Project Title: Classification of disaster-related tweets

This is a project to classify disaster-related tweets as disaster or non-disaster. The dataset contains around 7500 tweets that are labeled. The project uses NLP techniques to build a model that classifies the tweets.

## Project Structure

The project contains two datasets, train.csv and test.csv, which contain the tweet data. The helpers directory contains the code for data preparation and the implementation of machine learning models. The data_prepation.py script preprocesses the data, which involves removing URLs, mentions, stop words, and lemmatizing the text. The modeling.py script contains code to train the base models & ensembles to improve the predictions further. This script fits the CountVectorizer to the text data and trains a Naive Bayes and a Logistic Regression model. The basic_models.py script contains the code to fit the machine learning models. Finally, the word_embed-ktrain.py script uses the Ktrain library to train a BERT model to classify the tweets. Also, I used chatgpt-3.5turbo api to classify the tweets, and this approach performed poorly compared to the ktrain-bert approach. 


## Conclusion

The project successfully classifies the disaster-related tweets into disaster or non-disaster using NLP techniques and machine learning models. The F1 score is used to evaluate the models, and the best model has an F1 score of ~0.83
