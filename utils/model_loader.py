import pickle
from pathlib import Path


def load_model():
    """
    Load the pre-trained model and vectorizer

    Returns:
        tuple: (model, vectorizer) pair
    """
    model_path = Path(__file__).parent.parent / 'models' / 'fake_news_model.pkl'
    vectorizer_path = Path(__file__).parent.parent / 'models' / 'tfidf_vectorizer.pkl'

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer