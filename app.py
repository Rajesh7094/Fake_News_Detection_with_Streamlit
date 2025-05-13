import streamlit as st
import pickle
from pathlib import Path
from utils.preprocessing import preprocess_text

# Initialize model and vectorizer as global variables
model = None
vectorizer = None

# Download NLTK data
import nltk
import os
NLTK_DATA_DIR = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

# Set page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)


# Custom CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Using default styles.")


local_css("assets/style.css")


# Load model and vectorizer with error handling
@st.cache_resource
def load_model():
    global model, vectorizer  # Declare we're using the global variables
    try:
        model_path = Path('models/fake_news_model.pkl')
        vectorizer_path = Path('models/tfidf_vectorizer.pkl')

        if not model_path.exists():
            st.error(f"Model file not found at: {model_path.absolute()}")
            return None, None
        if not vectorizer_path.exists():
            st.error(f"Vectorizer file not found at: {vectorizer_path.absolute()}")
            return None, None

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)

        return model, vectorizer

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


# Load the model at startup
model, vectorizer = load_model()


# Main app
def main():
    global model, vectorizer  # Declare we're using the global variables

    st.title("📰 Fake News Detector")
    st.markdown("""
    This tool helps identify potentially fake news articles or social media posts using NLP techniques.
    Paste the text content below or upload a text file to analyze.
    """)

    # Check if model loaded successfully
    if model is None or vectorizer is None:
        st.error("""
        Model failed to load. Please check:
        1. Model files exist in 'models/' directory
        2. You've run the training script first
        3. Files are not corrupted
        """)

        if st.button("Try loading model again"):
            model, vectorizer = load_model()
            st.rerun()
        return

    # Rest of your main function remains the same...
    input_method = st.radio("Choose input method:", ("Paste Text", "Upload Text File"))

    text = ""
    if input_method == "Paste Text":
        text = st.text_area("Paste the news article content here:", height=200)
    else:
        uploaded_file = st.file_uploader("Upload a text file:", type=["txt"])
        if uploaded_file is not None:
            text = uploaded_file.getvalue().decode("utf-8")

    if st.button("Analyze") and text:
        with st.spinner("Analyzing content..."):
            # Preprocess text
            clean_text = preprocess_text(text)

            # Vectorize text
            text_vector = vectorizer.transform([clean_text])

            # Make prediction
            prediction = model.predict(text_vector)
            proba = model.predict_proba(text_vector)

            # Display results
            st.subheader("Analysis Results")

            col1, col2 = st.columns(2)
            with col1:
                if prediction[0] == 1:
                    st.error("🚨 Fake News Detected")
                else:
                    st.success("✅ Likely Real News")

            with col2:
                st.metric("Confidence Score",
                          f"{max(proba[0]) * 100:.2f}%")

            st.progress(max(proba[0]))

            # Show explanation
            st.subheader("Details")
            st.markdown(f"""
            - **Prediction:** {'Fake' if prediction[0] == 1 else 'Real'} news
            - **Confidence:** {max(proba[0]) * 100:.2f}%
            - **Text Length:** {len(text.split())} words
            """)

            # Show processed text (optional)
            with st.expander("View processed text"):
                st.write(clean_text)


if __name__ == "__main__":
    main()