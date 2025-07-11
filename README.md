import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# Sample dataset
data = {
    'review': [
        "I love this product! It's amazing.",
        "Worst experience ever. Totally useless.",
        "Not bad, but not great either.",
        "Excellent quality, very happy with it.",
        "Terrible experience. Waste of money."
    ],
    'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative']
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Pipeline: TF-IDF + Logistic Regression
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'sentiment_model.pkl')
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('sentiment_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    prediction = model.predict([review])[0]
    return render_template('index.html', review=review, sentiment=prediction)

if __name__ == '__main__':
    app.run(debug=True)

    <!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analyzer</title>
</head>
<body>
    <h2>Product Review Sentiment Classifier</h2>
    <form action="/predict" method="post">
        <textarea name="review" rows="4" cols="50" placeholder="Enter your product review here..."></textarea><br><br>
        <input type="submit" value="Predict">
    </form>

    {% if sentiment %}
        <h3>Review: {{ review }}</h3>
        <h3>Predicted Sentiment: {{ sentiment }}</h3>
    {% endif %}
</body>
</html>
