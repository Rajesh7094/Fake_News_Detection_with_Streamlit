import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
from utils.preprocessing import preprocess_text


def load_and_prepare_data():
    # Load datasets with error handling
    try:
        fake_df = pd.read_csv('data/Fake.csv')
        true_df = pd.read_csv('data/True.csv')
    except Exception as e:
        print(f"Error loading data files: {e}")
        print("Please ensure:")
        print("1. data/Fake.csv exists")
        print("2. data/True.csv exists")
        return None, None

    # Add labels
    fake_df['label'] = 1  # 1 for fake news
    true_df['label'] = 0  # 0 for true news

    # Combine title and text with NaN handling
    fake_df['text'] = fake_df['title'].fillna('') + " " + fake_df['text'].fillna('')
    true_df['text'] = true_df['title'].fillna('') + " " + true_df['text'].fillna('')

    # Combine datasets
    df = pd.concat([fake_df, true_df], axis=0)

    # Remove any empty texts
    df = df[df['text'].str.strip().astype(bool)]

    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)

    return df['text'], df['label']


def train_and_save_model():
    # Load and prepare data
    X, y = load_and_prepare_data()
    if X is None:
        return

    # Preprocess text with progress reporting
    print("Preprocessing text...")
    X_processed = X.progress_apply(preprocess_text)  # Requires tqdm: pip install tqdm

    # Remove empty processed texts
    valid_idx = X_processed.str.strip().astype(bool)
    X_processed = X_processed[valid_idx]
    y = y[valid_idx]

    # Create TF-IDF features
    print("\nCreating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)  # Use unigrams and bigrams
    )
    X_features = vectorizer.fit_transform(X_processed)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42
    )

    # Train model
    print("\nTraining model...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',  # Handle class imbalance
        C=0.1  # Regularization strength
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

    # Save model and vectorizer
    os.makedirs('models', exist_ok=True)
    with open('models/fake_news_model.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nModel and vectorizer saved to 'models' directory")


if __name__ == "__main__":
    # Add progress_apply if you want progress bars (requires tqdm)
    try:
        from tqdm import tqdm

        tqdm.pandas()
        pd.Series.progress_apply = pd.Series.progress_apply
    except ImportError:
        pd.Series.progress_apply = pd.Series.apply

    train_and_save_model()