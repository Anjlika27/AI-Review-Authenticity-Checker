"""
app.py — Flask backend for AI Review Authenticity Checker
Uses Logistic Regression for fast, memory-efficient deployment.
"""

from flask import Flask, render_template, request
import pandas as pd
import string
import threading
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

MODEL = None
MODEL_READY = False
MODEL_ERROR = None

def train_model():
    global MODEL, MODEL_READY, MODEL_ERROR
    try:
        print("Loading dataset...")
        df = pd.read_csv('dataset.csv')
        df.drop('Unnamed: 0', axis=1, inplace=True)
        df.dropna(inplace=True)
        print(f"Dataset loaded: {len(df)} rows")

        df['clean'] = df['text_'].apply(preprocess_text)

        X = df['clean']
        y = df['label']

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        print("Training Logistic Regression...")
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1, 2))),
            ('clf',   LogisticRegression(max_iter=500, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        MODEL = pipeline
        MODEL_READY = True
        print("Model trained and ready ✓")
    except Exception as e:
        MODEL_ERROR = str(e)
        print(f"Model training failed: {e}")

thread = threading.Thread(target=train_model)
thread.daemon = True
thread.start()

app = Flask(__name__)

@app.route('/')
def about():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    result = None
    review_text = ''

    if MODEL_ERROR:
        return render_template('prediction.html',
                               result=None, review_text='',
                               loading=False, error=MODEL_ERROR)

    if not MODEL_READY:
        return render_template('prediction.html',
                               result=None, review_text='',
                               loading=True, error=None)

    if request.method == 'POST':
        review_text = request.form.get('review', '').strip()
        if review_text:
            clean = preprocess_text(review_text)
            pred  = MODEL.predict([clean])[0]
            proba = MODEL.predict_proba([clean])[0]
            classes = MODEL.classes_.tolist()
            if pred == 'OR':
                label, css_class, emoji = 'GENUINE', 'result-genuine', '✅'
                confidence = proba[classes.index('OR')] * 100
            else:
                label, css_class, emoji = 'FAKE', 'result-fake', '🚨'
                confidence = proba[classes.index('CG')] * 100
            result = {
                'label':      label,
                'css_class':  css_class,
                'confidence': f'{confidence:.1f}',
                'emoji':      emoji,
            }

    return render_template('prediction.html',
                           result=result, review_text=review_text,
                           loading=False, error=None)

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
