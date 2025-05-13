#!/bin/bash
mkdir -p /home/adminuser/nltk_data
python -c "import nltk; nltk.download('punkt', download_dir='/home/adminuser/nltk_data'); nltk.download('stopwords', download_dir='/home/adminuser/nltk_data'); nltk.download('wordnet', download_dir='/home/adminuser/nltk_data')"