FROM python:3.11-slim

WORKDIR /app

# Definir variáveis de ambiente para o Streamlit
ENV HOME=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_ENABLE_STATIC_IP_INFO=false
ENV STREAMLIT_SERVER_FOLDER_CACHE_DIR=/tmp/streamlit_cache
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_BROWSER_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Adicionar um argumento de build para forçar a reconstrução do cache
ARG CACHE_BUST=3

# Usar o usuário root para garantir permissões de escrita
USER root

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Limpar o cache do pip e instalar as dependências
RUN pip cache purge && pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Instalar langchain-groq e langchain-community explicitamente
RUN pip install --no-cache-dir langchain-groq langchain-community

# Verificar se langchain-groq foi instalado
RUN pip show langchain-groq

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]
