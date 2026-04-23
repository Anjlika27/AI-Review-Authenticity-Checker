"""
app.py — Loads pre-trained model.pkl built during Render build step.
"""
import pickle
import string
import nltk
from nltk.corpus import stopwords
from flask import Flask, render_template, request

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

# preprocess_text MUST be defined here so pickle can find it
def preprocess_text(text):
    cleaned = ''.join(ch for ch in text if ch not in string.punctuation)
    return ' '.join(w for w in cleaned.split() if w.lower() not in STOP_WORDS)

print("Loading model...")
with open('model.pkl', 'rb') as f:
    MODEL = pickle.load(f)
print("Model loaded ✓")

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
                           result=result,
                           review_text=review_text,
                           loading=False,
                           error=None)

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
