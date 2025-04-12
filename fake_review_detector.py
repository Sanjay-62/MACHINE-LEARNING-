import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# Download NLTK resources
nltk.download('stopwords')

# Text preprocessing function
def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Load dataset (replace with your actual path or use st.file_uploader for online deployment)
df = pd.read_csv("Preprocessed Fake Reviews Detection Dataset.csv")
df.dropna(inplace=True)

# Split data
X = df['text_']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', LogisticRegression())
])

# Train model
pipeline.fit(X_train, y_train)

# Streamlit App
st.title("🕵️‍♂️ Fake Review Detector")
st.write("Enter a product review and we'll tell you if it's likely **Real** or **Fake**.")

user_review = st.text_area("Enter your review here:", "")

if st.button("Check Review"):
    if user_review.strip() == "":
        st.warning("Please enter a review.")
    else:
        prediction = pipeline.predict([user_review])[0]
        label = "✅ Real Review" if prediction == 1 else "❌ Fake Review"
        st.success(f"Prediction: {label}")
