import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import joblib
import argparse
import os

def evaluate_model(val_data_path, model_path, vectorizer_path, metrics_path):
    val_df = pd.read_csv(val_data_path)
    
    X_val = val_df['text']
    y_val = val_df['sentiment_id']
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    X_val_vectorized = vectorizer.transform(X_val)
    predictions = model.predict(X_val_vectorized)
    
    accuracy = accuracy_score(y_val, predictions)
    report = classification_report(y_val, predictions, output_dict=True)
    
    print(f"Precisión del modelo: {accuracy:.4f}")
    
    # Guardar métricas
    with open(metrics_path, 'w') as f:
        f.write(f"accuracy: {accuracy}\n")
        f.write(f"report: {report}\n")
    print(f"Métricas guardadas en {metrics_path}")

    # Aquí podrías añadir lógica para "validar" el modelo, por ejemplo, si la precisión es > 0.7
    if accuracy < 0.6: # Umbral de ejemplo muy bajo
        raise ValueError("El rendimiento del modelo es demasiado bajo para el despliegue.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--vectorizer_path', type=str, required=True)
    parser.add_argument('--metrics_path', type=str, required=True)
    args = parser.parse_args()
    
    evaluate_model(args.val_data_path, args.model_path, args.vectorizer_path, args.metrics_path)
