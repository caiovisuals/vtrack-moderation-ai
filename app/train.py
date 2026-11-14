import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Carregar dataset
data = pd.read_csv("data/dataset.csv")

texts = data["text"]
labels = data["label"]

# Vetorização
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Modelo
model = LogisticRegression()
model.fit(X, labels)

# Salvar modelo e vetor
joblib.dump(model, "app/model.joblib")
joblib.dump(vectorizer, "app/vectorizer.joblib")

print("Modelo treinado e salvo.")