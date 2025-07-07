import joblib
import pandas as pd
import os
import json
from google.cloud import storage

# Vertex AI espera una clase Predictor con load y predict
class CustomPredictor:
    def __init__(self):
        self._model = None
        self._vectorizer = None
        self._sentiment_mapping_rev = None

    def load(self, model_dir):
        """Carga el modelo y sus artefactos auxiliares del directorio."""
        self._model = joblib.load(os.path.join(model_dir, 'sentiment_model.joblib'))
        self._vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.joblib'))
        
        self._sentiment_mapping_rev = {}
        with open(os.path.join(model_dir, 'sentiment_mapping.txt'), 'r') as f:
            for line in f:
                sentiment, id_val = line.strip().split(',')
                self._sentiment_mapping_rev[int(id_val)] = sentiment
        print(f"Modelo, vectorizador y mapeo cargados de {model_dir}")

    def predict(self, instances):
        """Realiza predicciones sobre las instancias de entrada."""
        # instances es una lista de diccionarios, cada uno con una clave 'text'
        texts = [instance['text'] for instance in instances]
        
        vectorized_texts = self._vectorizer.transform(texts)
        predictions_ids = self._model.predict(vectorized_texts)
        
        predictions_sentiment = [self._sentiment_mapping_rev[id_val] for id_val in predictions_ids]
        
        # Vertex AI espera una lista de diccionarios como respuesta
        return [{"prediction": sentiment} for sentiment in predictions_sentiment]

# Para la compatibilidad con Vertex AI, no hay un `if __name__ == '__main__'` aquí para la lógica de serving.
# Vertex AI se encarga de instanciar la clase `CustomPredictor` y llamar a `load()` y `predict()`.
