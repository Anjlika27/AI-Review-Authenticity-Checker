"""
app.py — Flask backend for AI Review Authenticity Checker
Model trains in a background thread so the port binds immediately.
"""

from flask import Flask, render_template, request
import pandas as pd
import string
import threading
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

# Global model variable
MODEL = None
MODEL_READY = False

def train_model():
    global MODEL, MODEL_READY
    print("Training SVM model in background...")
    df = pd.read_csv('dataset.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.dropna(inplace=True)

    X = df['text_']
    y = df['label']

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('bow',   CountVectorizer(analyzer=preprocess_text)),
        ('tfidf', TfidfTransformer()),
        ('clf',   SVC(kernel='linear', C=1.0, probability=True, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    MODEL = pipeline
    MODEL_READY = True
    print("Model trained and ready ✓")

# Start training in background thread
thread = threading.Thread(target=train_model)
thread.daemon = True
thread.start()

# ── Flask app ─────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route('/')
def about():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    result = None
    review_text = ''

    if not MODEL_READY:
        return render_template('prediction.html',
                               result=None,
                               review_text='',
                               loading=True)

    if request.method == 'POST':
        review_text = request.form.get('review', '').strip()
        if review_text:
            pred  = MODEL.predict([review_text])[0]
            proba = MODEL.predict_proba([review_text])[0]
            if pred == 'OR':
                label, css_class, confidence, emoji = (
                    'GENUINE', 'result-genuine', proba[1] * 100, '✅'
                )
            else:
                label, css_class, confidence, emoji = (
                    'FAKE', 'result-fake', proba[0] * 100, '🚨'
                )
            result = {
                'label':      label,
                'css_class':  css_class,
                'confidence': f'{confidence:.1f}',
                'emoji':      emoji,
            }

    return render_template('prediction.html',
                           result=result,
                           review_text=review_text,
                           loading=False)

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
