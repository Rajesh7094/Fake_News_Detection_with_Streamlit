import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """
    Clean and preprocess text for fake news detection

    Args:
        text (str): Raw input text

    Returns:
        str: Cleaned and processed text
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    try:
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize with error handling
        tokens = word_tokenize(text)

        # Remove stopwords
        tokens = [t for t in tokens if t not in stop_words]

        # Lemmatization
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

        # Join tokens back to string
        clean_text = ' '.join(tokens)

        return clean_text

    except Exception as e:
        print(f"Error processing text: {e}")
        return ""