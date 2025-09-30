import os
import subprocess
import threading
import time
from flask import Flask, redirect
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Variável global para controlar o processo do Streamlit
streamlit_process = None

def start_streamlit():
    """Inicia o servidor Streamlit em uma thread separada"""
    global streamlit_process
    try:
        # Define a variável de ambiente para a chave da API
        env = os.environ.copy()
        env['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', '')
        
        # Inicia o Streamlit na porta 8501
        streamlit_process = subprocess.Popen([
            'streamlit', 'run', 'app.py', 
            '--server.port=8501',
            '--server.address=0.0.0.0',
            '--server.headless=true',
            '--browser.gatherUsageStats=false'
        ], env=env)
        
        print("Streamlit iniciado com sucesso!")
    except Exception as e:
        print(f"Erro ao iniciar Streamlit: {e}")

@app.route('/')
def index():
    """Redireciona para a aplicação Streamlit"""
    return redirect('http://localhost:8501')

@app.route('/health')
def health():
    """Endpoint de saúde para verificar se a aplicação está funcionando"""
    return {"status": "healthy", "streamlit_running": streamlit_process is not None}

if __name__ == '__main__':
    # Inicia o Streamlit em uma thread separada
    streamlit_thread = threading.Thread(target=start_streamlit)
    streamlit_thread.daemon = True
    streamlit_thread.start()
    
    # Aguarda um pouco para o Streamlit iniciar
    time.sleep(3)
    
    # Inicia o servidor Flask
    app.run(host='0.0.0.0', port=5000, debug=False)
