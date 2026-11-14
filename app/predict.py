import joblib

model = joblib.load("app/model.joblib")
vectorizer = joblib.load("app/vectorizer.joblib")

def predict_text(text: str):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {
        "label": int(prediction),
        "score": float(probability)
    }