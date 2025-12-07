import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Run once (manually) before first use
# nltk.download('punkt')
# nltk.download('stopwords')

stop_words = set(stopwords.words("english"))

def clean_text(text):
    """
    Cleans raw text extracted from PDF or user input.
    Removes numbers, punctuation, stopwords, etc.
    """

    if not text or not isinstance(text, str):
        return ""

    # 1. Lowercase the text
    text = text.lower()

    # 2. Remove weird PDF characters (like \n, \t, bullets)
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces & newlines

    # 3. Remove numbers
    text = re.sub(r'\d+', '', text)

    # 4. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 5. Tokenize (split)
    tokens = word_tokenize(text)

    # 6. Remove stopwords
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    # 7. Join back to sentence
    clean = " ".join(tokens)

    return clean


def preprocess_text(text):
    """
    Final preprocess function used in other scripts.
    """
    return clean_text(text)
