import pandas as pd
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
from nltk.corpus import stopwords

df = pd.read_csv('data/train.csv')


def preprocessing_text(df):
    """
    Preprocess the text data by removing non-alphabetic characters, urls and mentions
    """
    # filter out @mentions and URLs as a separate column from the "text" column in df
    df['text'] = df['text'].str.replace(r'@\w+', '')
    df['text'] = df['text'].str.replace(r'http\S+', '')
    df['text'] = df['text'].str.replace(r'www\S+', '')
    df['text'] = df['text'].str.replace(r'pic.twitter.com\S+', '')
    df['text'] = df['text'].str.replace(r'pic.twitter\S+', '')

    # encode all text that has encoding starting with \
    df['text'] = df['text'].str.encode('ascii', 'ignore').str.decode('ascii')

    # remove all non-ascii characters
    df['text'] = df['text'].str.replace(r'[^\x00-\x7F]+', '')

    # remove all non-alphanumeric characters
    df['text'] = df['text'].str.replace(r'[^a-zA-Z0-9\s]', '')

    # remove all single characters
    # df['text'] = df['text'].str.replace(r'\b[a-zA-Z]\b', '')

    # trim all leading and trailing whitespaces
    df['text'] = df['text'].str.strip()

    # remove all whitespaces
    df['text'] = df['text'].str.replace(r'\s+', ' ')

    # to lowercase
    df['text'] = df['text'].str.lower()

    return df


def lemmatize(df):
    """
    Process the text data using nltk library and
    lemmatize the words to their root form
    :return:
    """
    lemmatizer = WordNetLemmatizer()
    df['text'] = df['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    return df


def remove_stopwords(df):
    """
    Remove stopwords from the text data
    :return:
    """
    stop_words = set(stopwords.words('english'))
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    return df


df = preprocessing_text(df)
df = lemmatize(df)
df = remove_stopwords(df)

a = 5