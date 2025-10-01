FROM python:3.11-slim

WORKDIR /app

# Definir a variável HOME para um diretório gravável
ENV HOME=/app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Copiar o config.toml para o diretório .streamlit dentro do HOME
RUN mkdir -p ${HOME}/.streamlit && cp .streamlit/config.toml ${HOME}/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
