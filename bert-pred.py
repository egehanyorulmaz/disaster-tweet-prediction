import ktrain
import pandas as pd

# Load the model
predictor = ktrain.load_predictor('models/bert')

## Prediction
df = pd.read_csv('data/test_preprocessed.csv')
ids = df['id']
df = df.drop(columns=['id', 'location'])
df.fillna('no_value', inplace=True)

predictions = predictor.predict(df['text'].tolist())

df = pd.DataFrame({'id': ids, 'target': predictions})
df.to_csv('data/submission_bert.csv', index=False)

a = 5
