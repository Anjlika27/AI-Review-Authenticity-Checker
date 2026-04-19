"""
app.py — Flask backend for AI Review Authenticity Checker
preprocess_text MUST be defined here before pickle.load()
because the SVM pipeline was serialised with this function as its analyzer.
"""

from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

# ── Text preprocessor (must match the one used during training) ───────
STOP_WORDS = set(stopwords.words('english'))

def preprocess_text(text):
    """Remove punctuation and stopwords, return token list."""
    cleaned = ''.join(ch for ch in text if ch not in string.punctuation)
    return [w for w in cleaned.split() if w.lower() not in STOP_WORDS]

# ── Load the SVM model (pickle can now find preprocess_text) ──────────
with open('svm_model.pkl', 'rb') as f:
    MODEL = pickle.load(f)

app = Flask(__name__)

# ── Routes ────────────────────────────────────────────────────────────

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

    return render_template('prediction.html',
                           result=result,
                           review_text=review_text)


@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)
