import pickle
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import math
pd.options.display.float_format = '{:,.2f}'.format
sns.set_style("whitegrid")


def save_model(model, model_path):
    """
    Save the model to a file
    """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(model_path):
    """
    Load the model from a file
    """
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def eda_categorical(data, variable, ax=None):
    variable_df = data[variable].value_counts(normalize=True).reset_index()
    n_colors = len(variable_df)
    variable_df.set_index('index').T.plot(kind='barh',
                                          stacked=True,
                                          colormap=ListedColormap(sns.color_palette("Set2", n_colors)),
                                          width=0.15, ax=ax)


def multiple_eda_categorical(data, list_categorical):
    n_rows = math.ceil(len(list_categorical) / 2)
    fig = plt.figure(figsize=(12, n_rows * 3))

    for i, variable in enumerate(list_categorical):
        ax = fig.add_subplot(n_rows, 2, i + 1)
        eda_categorical(data, variable, ax=ax)

    plt.tight_layout()
    plt.show()


def multiple_eda_continuous(data, list_continuous):
    n_rows = math.ceil(len(list_continuous) / 3)
    fig = plt.figure(figsize=(12, n_rows * 5))
    palette = sns.color_palette('Set2', 3)

    for i, variable in enumerate(list_continuous):
        ax = fig.add_subplot(n_rows, 3, i + 1)
        sns.boxplot(x=variable, data=data, orient='v', palette=[palette[i]], ax=ax)
        ax.set_ylabel('')
        ax.set_title(variable)

    plt.tight_layout()
    plt.show()