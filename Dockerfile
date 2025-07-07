# Utiliza una imagen base de Python oficial
FROM python:3.9-slim-buster

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia los archivos de requerimientos y los instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia los scripts de la aplicación
COPY src/ ./src/
COPY data/ ./data/ # Copia datos si son necesarios para algún paso dentro del contenedor

# Comando para el contenedor (no es necesario un CMD para los componentes del pipeline, ya que kfp los invoca)
