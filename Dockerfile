# Usar Python 3.11 tal y como requieren tus dependencias (langchain-neo4j)
FROM python:3.11-slim

# Directorio de trabajo en el contenedor
WORKDIR /app

# Instalar dependencias del sistema necesarias para compilar algunas librerías
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY . .

# Exponer el puerto de Streamlit
EXPOSE 8501

# Comando de arranque del frontend
CMD ["streamlit", "run", "src/frontend_app.py", "--server.address=0.0.0.0"]