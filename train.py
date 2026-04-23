"""
train.py — Run during Render build step to create model.pkl
"""
import pandas as pd
import string
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

def preprocess_text(text):
    cleaned = ''.join(ch for ch in text if ch not in string.punctuation)
    return ' '.join(w for w in cleaned.split() if w.lower() not in STOP_WORDS)

print("Loading dataset...")
df = pd.read_csv('dataset.csv')

# Drop unnamed index column only if it exists
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

df.dropna(inplace=True)
print(f"Dataset: {len(df)} rows")
print(f"Labels: {df['label'].value_counts().to_dict()}")

df['clean'] = df['text_'].apply(preprocess_text)
X = df['clean']
y = df['label']

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("Training model...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1, 2))),
    ('clf',   LogisticRegression(max_iter=500, random_state=42))
])
pipeline.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("model.pkl saved ✓")
