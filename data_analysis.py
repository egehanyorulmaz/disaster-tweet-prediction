from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from nltk.tokenize import RegexpTokenizer
import pandas as pd

tokenizer = RegexpTokenizer(r'\w+')

clean_questions = pd.read_csv("data/train_preprocessed.csv")

clean_questions["tokens"] = clean_questions["text"].apply(tokenizer.tokenize)

all_words = [word for tokens in clean_questions["tokens"] for word in tokens]
sentence_lengths = [len(tokens) for tokens in clean_questions["tokens"]]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
print("Max sentence length is %s" % max(sentence_lengths))