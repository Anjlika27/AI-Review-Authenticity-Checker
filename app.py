"""
app.py — Flask backend for AI Review Authenticity Checker
Trains the SVM model on startup (no pickle needed).
"""

from flask import Flask, render_template, request
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

nltk.download('stopwords', quiet=True)

# ── Preprocessor ──────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words('english'))

def preprocess_text(text):
    cleaned = ''.join(ch for ch in text if ch not in string.punctuation)
    return [w for w in cleaned.split() if w.lower() not in STOP_WORDS]

# ── Train model on startup ────────────────────────────────────────────
print("Loading dataset and training SVM model...")
df = pd.read_csv('dataset.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df.dropna(inplace=True)

X = df['text_']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

MODEL = Pipeline([
    ('bow',   CountVectorizer(analyzer=preprocess_text)),
    ('tfidf', TfidfTransformer()),
    ('clf',   SVC(kernel='linear', C=1.0, probability=True, random_state=42))
])
MODEL.fit(X_train, y_train)
print("Model trained and ready ✓")

# ── Flask app ─────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route('/')
def about():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    result = None
    review_text = ''
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
    return render_template('prediction.html', result=result, review_text=review_text)

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
