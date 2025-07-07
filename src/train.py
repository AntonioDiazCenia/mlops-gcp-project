import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import argparse
import os

def train_model(train_data_path, model_dir):
    train_df = pd.read_csv(train_data_path)
    
    X_train = train_df['text']
    y_train = train_df['sentiment_id']
    
    # Vectorización TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    # Entrenamiento del modelo
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vectorized, y_train)
    
    # Guardar modelo y vectorizador
    model_path = os.path.join(model_dir, 'sentiment_model.joblib')
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"Modelo entrenado y guardado en {model_path}")
    print(f"Vectorizer guardado en {vectorizer_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.model_dir, exist_ok=True)
    train_model(args.train_data_path, args.model_dir)
