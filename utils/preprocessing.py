import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Use a writable directory in the app's workspace
NLTK_DATA_DIR = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

# Download required NLTK data if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=NLTK_DATA_DIR)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir=NLTK_DATA_DIR)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=NLTK_DATA_DIR)

# Initialize components after ensuring data is available
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
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