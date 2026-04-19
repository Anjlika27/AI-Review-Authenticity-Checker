"""
app.py — Flask backend for AI Review Authenticity Checker
Uses LinearSVC (much faster + lighter than SVC) for deployment.
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
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))

def preprocess_text(text):
    cleaned = ''.join(ch for ch in text if ch not in string.punctuation)
    return [w for w in cleaned.split() if w.lower() not in STOP_WORDS]

# Global model variable
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

        # Use only 60% of data to save memory on free tier
        df = df.sample(frac=0.6, random_state=42).reset_index(drop=True)
        print(f"Training on {len(df)} samples...")

        X = df['text_']
        y = df['label']

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # LinearSVC is 10x faster and uses much less memory than SVC
        pipeline = Pipeline([
            ('bow',   CountVectorizer(analyzer=preprocess_text, max_features=20000)),
            ('tfidf', TfidfTransformer()),
            ('clf',   CalibratedClassifierCV(LinearSVC(max_iter=1000)))
        ])
        pipeline.fit(X_train, y_train)
        MODEL = pipeline
        MODEL_READY = True
        print("Model trained and ready ✓")
    except Exception as e:
        MODEL_ERROR = str(e)
        print(f"Model training failed: {e}")

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

    if MODEL_ERROR:
        return render_template('prediction.html',
                               result=None,
                               review_text='',
                               loading=False,
                               error=MODEL_ERROR)

    if not MODEL_READY:
        return render_template('prediction.html',
                               result=None,
                               review_text='',
                               loading=True,
                               error=None)

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
                           loading=False,
                           error=None)

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
