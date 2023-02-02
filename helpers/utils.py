import pickle


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
