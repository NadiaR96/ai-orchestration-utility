# utils/setup_nltk.py
import nltk

def download_nltk_resources():
    resources = ["punkt"]
    for res in resources:
        nltk.download(res, quiet=True)
    print("✅ NLTK resources downloaded successfully!")

if __name__ == "__main__":
    download_nltk_resources()