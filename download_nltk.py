import nltk

def download_required_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')  # This is the missing resource
    nltk.download('omw-1.4')    # Needed for WordNet lemmatization

if __name__ == "__main__":
    download_required_data()
    print("All required NLTK data downloaded successfully!")