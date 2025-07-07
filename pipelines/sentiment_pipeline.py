from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import (
    component, 
    Artifact, 
    Input, 
    Output, 
    Model, 
    Dataset, 
    Metrics
)
from google.cloud import aiplatform

# --- Configuración global ---
PROJECT_ID = 'your-project-id' # Reemplaza con tu Project ID
REGION = 'your-region'       # Reemplaza con tu región (ej: us-central1)
GCS_BUCKET = 'your-unique-bucket-name' # Reemplaza con el nombre de tu bucket
PIPELINE_ROOT = f'gs://{GCS_BUCKET}/pipeline_root'

# --- 1. Definición de Componentes (basados en Docker) ---
# Puedes definir los componentes directamente desde el código fuente Python si el Dockerfile ya está configurado.
# Esto asume que tienes la imagen Docker base construida y disponible en Artifact Registry.
# Para este ejemplo, usaremos el mismo Dockerfile para todos, lo que simplifica.
# En un escenario real, podrías tener Dockerfiles específicos para cada componente.

# Imagen base para los componentes del pipeline
# Asegúrate de que esta imagen exista en Artifact Registry:
# gcr.io/your-project-id/mlops-pipeline-base:latest
PIPELINE_CONTAINER_IMAGE = f"gcr.io/{PROJECT_ID}/mlops-pipeline-base:latest"

@component(
    base_image=PIPELINE_CONTAINER_IMAGE,
    output_component_file="preprocess_component.yaml"
)
def preprocess_data_component(
    input_data_uri: str,
    training_data: Output[Dataset],
    validation_data: Output[Dataset],
    sentiment_mapping: Output[Artifact]
):
    """Preprocesses raw data and splits it into training and validation sets."""
    return dsl.ContainerSpec(
        image=PIPELINE_CONTAINER_IMAGE,
        command=[
            'python', 'src/preprocess.py',
            '--input_data_path', input_data_uri,
            '--output_dir', './processed_data'
        ],
        outputs={
            'training_data': {'path': './processed_data/train_data.csv'},
            'validation_data': {'path': './processed_data/val_data.csv'},
            'sentiment_mapping': {'path': './processed_data/sentiment_mapping.txt'}
        }
    )

@component(
    base_image=PIPELINE_CONTAINER_IMAGE,
    output_component_file="train_component.yaml"
)
def train_model_component(
    training_data: Input[Dataset],
    model: Output[Model],
    vectorizer: Output[Artifact]
):
    """Trains the sentiment classification model."""
    return dsl.ContainerSpec(
        image=PIPELINE_CONTAINER_IMAGE,
        command=[
            'python', 'src/train.py',
            '--train_data_path', training_data.path,
            '--model_dir', './model_output'
        ],
        outputs={
            'model': {'path': './model_output/sentiment_model.joblib'},
            'vectorizer': {'path': './model_output/tfidf_vectorizer.joblib'}
        }
    )

@component(
    base_image=PIPELINE_CONTAINER_IMAGE,
    output_component_file="evaluate_component.yaml"
)
def evaluate_model_component(
    validation_data: Input[Dataset],
    model: Input[Model],
    vectorizer: Input[Artifact],
    metrics: Output[Metrics]
):
    """Evaluates the trained model."""
    return dsl.ContainerSpec(
        image=PIPELINE_CONTAINER_IMAGE,
        command=[
            'python', 'src/evaluate.py',
            '--val_data_path', validation_data.path,
            '--model_path', model.path,
            '--vectorizer_path', vectorizer.path,
            '--metrics_path', metrics.path
        ],
        outputs={
            'metrics': {'path': metrics.path}
        }
    )

# --- 2. Definición del Pipeline ---

@dsl.pipeline(
    name="sentiment-analysis-mlops-pipeline",
    description="An MLOps pipeline for sentiment analysis using Vertex AI.",
    pipeline_root=PIPELINE_ROOT
)
def sentiment_analysis_pipeline(
    raw_data_gcs_uri: str
):
    # Paso 1: Preprocesamiento de datos
    preprocess_task = preprocess_data_component(
        input_data_uri=raw_data_gcs_uri
    )

    # Paso 2: Entrenamiento del modelo
    train_task = train_model_component(
        training_data=preprocess_task.outputs['training_data']
    )

    # Paso 3: Evaluación del modelo
    evaluate_task = evaluate_model_component(
        validation_data=preprocess_task.outputs['validation_data'],
        model=train_task.outputs['model'],
        vectorizer=train_task.outputs['vectorizer']
    )

    # Paso 4: Despliegue condicional del modelo (si las métricas son buenas)
    # Vertex AI Model Registry y Endpoint Deployment
    with dsl.Condition(evaluate_task.outputs['metrics'].metadata['accuracy'] > 0.6): # Umbral de ejemplo
        model_upload_op = aiplatform.ModelUploadOp(
            project=PROJECT_ID,
            display_name="sentiment-analysis-model",
            unmanaged_container_model=aiplatform.UnmanagedContainerModel(
                model_path=train_task.outputs['model'].uri,
                artifact_uri_vectorizer=train_task.outputs['vectorizer'].uri, # Pasar artefacto del vectorizador
                artifact_uri_sentiment_map=preprocess_task.outputs['sentiment_mapping'].uri, # Pasar artefacto del mapeo
                serving_container_image_uri=f"gcr.io/{PROJECT_ID}/sentiment-model-serving:latest",
                serving_container_predict_route="/predict",
                serving_container_health_route="/health",
                serving_container_args=[
                    "--model_dir", model.uri,
                    "--vectorizer_path", vectorizer.uri,
                    "--sentiment_mapping_path", sentiment_mapping.uri # Pasar al contenedor de serving
                ]
            )
        )
        model_upload_op.after(evaluate_task) # Asegurarse de que se ejecuta después de la evaluación

        model_deploy_op = aiplatform.ModelDeployOp(
            project=PROJECT_ID,
            model=model_upload_op.outputs['model'],
            endpoint_display_name="sentiment-analysis-endpoint",
            machine_type="n1-standard-2", # Ajusta según necesidad
            min_replica_count=1,
            max_replica_count=1
        )
        model_deploy_op.after(model_upload_op)

        # Configurar monitoreo del modelo (requiere el endpoint desplegado)
        # Esto es más complejo y generalmente se hace fuera del pipeline de entrenamiento directo,
        # pero se puede integrar. Por simplicidad, solo mencionaremos la capacidad.
        # aiplatform.ModelMonitoringJobCreateOp(...)


# --- 3. Compilación del Pipeline ---
# Este script se compila para generar un JSON de pipeline que Cloud Build ejecutará.
if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=sentiment_analysis_pipeline,
        package_path="sentiment_analysis_pipeline.json"
    )
    print("Pipeline compilado a sentiment_analysis_pipeline.json")
