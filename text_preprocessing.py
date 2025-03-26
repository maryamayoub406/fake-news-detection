import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()  
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization
    return " ".join(words)

# Load dataset
df = pd.read_csv("dataset/cleaned_data.csv")

# Apply text cleaning
df["text"] = df["title"] + " " + df["text"]
df["text"] = df["text"].apply(clean_text)

# Save preprocessed data
df.to_csv("dataset/preprocessed_data.csv", index=False)
