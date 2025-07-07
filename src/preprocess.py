import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

def preprocess_data(input_path, output_dir):
    df = pd.read_csv(input_path)
    
    # Simula un preprocesamiento simple: mapeo de sentimientos a números
    sentiment_mapping = {'positive': 0, 'negative': 1, 'neutral': 2}
    df['sentiment_id'] = df['sentiment'].map(sentiment_mapping)
    
    # Guardar mapeo (para usar en serving)
    with open(os.path.join(output_dir, 'sentiment_mapping.txt'), 'w') as f:
        for sentiment, id_val in sentiment_mapping.items():
            f.write(f"{sentiment},{id_val}\n")

    # Dividir datos
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_output_path = os.path.join(output_dir, 'train_data.csv')
    val_output_path = os.path.join(output_dir, 'val_data.csv')
    
    train_df.to_csv(train_output_path, index=False)
    val_df.to_csv(val_output_path, index=False)
    
    print(f"Datos preprocesados y divididos. Entrenamiento en {train_output_path}, Validación en {val_output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    preprocess_data(args.input_data_path, args.output_dir)
