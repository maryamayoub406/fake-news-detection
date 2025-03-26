from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import joblib  # Using Joblib instead of Pickle
import pandas as pd

# Load dataset
df = pd.read_csv("dataset/preprocessed_data.csv")

# Check class distribution
print("📊 Class distribution before balancing:\n", df["label"].value_counts())

# ✅ Step 1: Balance Dataset (if needed)
fake_news = df[df["label"] == 0]
real_news = df[df["label"] == 1]

if len(fake_news) < len(real_news):  # If Fake News is less, balance it
    fake_news = resample(fake_news, replace=True, n_samples=len(real_news), random_state=42)
    df = pd.concat([real_news, fake_news])

# Shuffle data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Check new class distribution
print("✅ Class distribution after balancing:\n", df["label"].value_counts())

# ✅ Step 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# ✅ Step 3: Convert text to numerical data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ✅ Step 4: Train model using Naïve Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ✅ Step 5: Evaluate model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Model Accuracy:", accuracy)
print("\n🔍 Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Step 6: Save model and vectorizer
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n✅ Model & Vectorizer saved successfully in 'models/' directory!")
