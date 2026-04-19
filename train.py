"""
train.py — Trains the SVM model and saves it as svm_model.pkl
Run this once before starting the Flask app, or call it in the build command.
"""

import pandas as pd
import string
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))

def preprocess_text(text):
    cleaned = ''.join(ch for ch in text if ch not in string.punctuation)
    return [w for w in cleaned.split() if w.lower() not in STOP_WORDS]

print("Loading dataset...")
df = pd.read_csv('dataset.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df.dropna(inplace=True)

X = df['text_']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("Training SVM model...")
pipeline = Pipeline([
    ('bow',   CountVectorizer(analyzer=preprocess_text)),
    ('tfidf', TfidfTransformer()),
    ('clf',   SVC(kernel='linear', C=1.0, probability=True, random_state=42))
])

pipeline.fit(X_train, y_train)

with open('svm_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model saved as svm_model.pkl ✓")
