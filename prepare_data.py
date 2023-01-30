import pandas as pd
from helpers.data_preparation import preprocessing_pipeline

args = {
    'frequency_threshold': 750,
    'train': True
}
print("Arguments: ", args)
df = pd.read_csv('data/train.csv')
print('Preprocessing the data...')
preprocessor = preprocessing_pipeline(args)
df = preprocessor.fit_transform(df)
df.to_csv('data/train_preprocessed.csv', index=False)

## processing test data
args = {
    'frequency_threshold': 750,
    'train': False
}
print("Arguments: ", args)
df = pd.read_csv('data/test.csv')
print('Preprocessing the data...')
preprocessor = preprocessing_pipeline(args)
df = preprocessor.fit_transform(df)
df.to_csv('data/test_preprocessed.csv', index=False)
print("Preprocessing done!")