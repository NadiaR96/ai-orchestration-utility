# utils/setup_nltk.py
import nltk

def download_nltk_resources():
    # Newer NLTK tokenizers may require both punkt and punkt_tab.
    resources = ["punkt", "punkt_tab"]
    for res in resources:
        nltk.download(res, quiet=True)
    print("✅ NLTK resources downloaded successfully!")

if __name__ == "__main__":
    download_nltk_resources()