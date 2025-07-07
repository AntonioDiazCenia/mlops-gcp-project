import joblib
import pandas as pd
import os
import json

class CustomPredictor:
    def __init__(self, model_dir):
        self._model = joblib.load(os.path.join(model_dir, 'sentiment_model.joblib'))
        self._vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.joblib'))
        
        # Cargar el mapeo de sentimientos
        self._sentiment_mapping_rev = {}
        with open(os.path.join(model_dir, 'sentiment_mapping.txt'), 'r') as f:
            for line in f:
                sentiment, id_val = line.strip().split(',')
                self._sentiment_mapping_rev[int(id_val)] = sentiment

    def predict(self, instances):
        # instances es una lista de strings (textos)
        texts = [instance['text'] for instance in instances]
        
        vectorized_texts = self._vectorizer.transform(texts)
        predictions_ids = self._model.predict(vectorized_texts)
        
        # Mapear los IDs de vuelta a los sentimientos
        predictions_sentiment = [self._sentiment_mapping_rev[id_val] for id_val in predictions_ids]
        
        return [{"prediction": sentiment} for sentiment in predictions_sentiment]

# Para pruebas locales o cuando Vertex AI carga el predictor
if __name__ == '__main__':
    # Simular la carga del modelo para pruebas
    # Asegúrate de tener los archivos model_sentiment.joblib, tfidf_vectorizer.joblib, sentiment_mapping.txt en un directorio 'local_model_dir'
    
    # Crear un directorio de modelo local para la prueba
    os.makedirs('local_model_dir', exist_ok=True)
    
    # Puedes copiar los archivos de modelo y vectorizador aquí si los entrenaste localmente
    # Si no, esto es solo un placeholder para la estructura.
    # Por ejemplo, correr preprocess.py y train.py localmente para generar estos archivos.
    
    # Ejemplo de uso:
    predictor = CustomPredictor('local_model_dir')
    test_instances = [{"text": "This movie was fantastic!"}, {"text": "I hated the food."}]
    results = predictor.predict(test_instances)
    print(json.dumps(results, indent=2))
