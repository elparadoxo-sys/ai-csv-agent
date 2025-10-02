import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

def main():
    st.set_page_config(
        page_title="🤖 Agente de Análise de CSV",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Agente de Análise de CSV com LangChain")
    st.markdown("""
    Esta aplicação utiliza um agente de IA baseado em **LangChain** para analisar arquivos CSV.
    Faça upload de um arquivo CSV e faça perguntas sobre seus dados!
    """)

    # Verificar se a chave da API está configurada
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("⚠️ A chave da API da Groq não foi configurada. Configure a variável de ambiente GROQ_API_KEY.")
        st.stop()

    # Sidebar para upload de arquivo
    with st.sidebar:
        st.header("📁 Upload de Arquivo")
        uploaded_file = st.file_uploader(
            "Escolha um arquivo CSV",
            type="csv",
            help="Faça upload de um arquivo CSV para análise"
        )
        
        if uploaded_file is not None:
            try:
                # Carregar o DataFrame
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Arquivo carregado com sucesso!")
                st.info(f"📊 **Dimensões:** {df.shape[0]} linhas × {df.shape[1]} colunas")
                
                # Mostrar preview dos dados
                st.subheader("👀 Preview dos Dados")
                st.dataframe(df.head(), use_container_width=True)
                
                # Informações básicas sobre o dataset
                st.subheader("📈 Informações Básicas")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Linhas", df.shape[0])
                    st.metric("Colunas", df.shape[1])
                with col2:
                    st.metric("Valores Nulos", df.isnull().sum().sum())
                    st.metric("Memória (MB)", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}")
                
                # Armazenar o DataFrame no session state
                st.session_state.df = df
                st.session_state.file_uploaded = True
                
            except Exception as e:
                st.error(f"❌ Erro ao carregar o arquivo: {str(e)}")
                st.session_state.file_uploaded = False

    # Área principal da aplicação
    if 'file_uploaded' in st.session_state and st.session_state.file_uploaded:
        
        # Inicializar histórico de conversa
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Mostrar histórico de mensagens
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and "plot" in message:
                    st.pyplot(message["plot"])
                else:
                    st.markdown(message["content"])
        
        # Input do usuário
        if prompt := st.chat_input("Faça uma pergunta sobre os dados..."):
            # Adicionar mensagem do usuário
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Gerar resposta do agente
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analisando os dados..."):
                    try:
                        # Criar o agente LangChain
                        llm = ChatGroq(temperature=0, groq_api_key=groq_api_key)
                        agent = create_pandas_dataframe_agent(
                            llm, 
                            st.session_state.df, 
                            verbose=False,
                            allow_dangerous_code=True
                        )
                        
                        # Executar a consulta
                        response = agent.run(prompt)
                        
                        # Exibir a resposta
                        st.markdown(response)
                        
                        # Adicionar ao histórico
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response
                        })
                        
                    except Exception as e:
                        error_msg = f"❌ Ocorreu um erro: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })
        
        # Seção de exemplos de perguntas
        with st.expander("💡 Exemplos de Perguntas"):
            st.markdown("""
            **Análise Descritiva:**
            - Quais são os tipos de dados de cada coluna?
            - Mostre as estatísticas descritivas dos dados
            - Quantos valores únicos existem em cada coluna?
            
            **Visualizações:**
            - Crie um histograma da coluna [nome_da_coluna]
            - Mostre a correlação entre as variáveis numéricas
            - Faça um gráfico de dispersão entre [coluna1] e [coluna2]
            
            **Análise de Padrões:**
            - Existem outliers nos dados?
            - Qual é a distribuição da variável target?
            - Quais são os padrões mais interessantes nos dados?
            """)
    
    else:
        # Tela inicial quando nenhum arquivo foi carregado
        st.info("👆 Faça upload de um arquivo CSV na barra lateral para começar a análise!")
        
        # Informações sobre a aplicação
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🧠 Inteligência Artificial")
            st.write("Utiliza LangChain e Groq para análise inteligente de dados")
        
        with col2:
            st.subheader("📊 Análise Completa")
            st.write("Estatísticas descritivas, visualizações e detecção de padrões")
        
        with col3:
            st.subheader("💬 Interface Conversacional")
            st.write("Faça perguntas em linguagem natural sobre seus dados")

if __name__ == "__main__":
    main()
