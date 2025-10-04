import streamlit as st
import pandas as pd
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
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
                # Salvar o arquivo CSV temporariamente
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Carregar o CSV para um DataFrame (apenas para preview e info básicas)
                # Carrega apenas algumas linhas para preview e informações básicas para evitar estouro de memória
                df_preview = pd.read_csv(tmp_file_path, nrows=5)
                
                # Para obter o número total de linhas e colunas sem carregar tudo na memória
                # Lendo o arquivo em chunks para contar linhas e colunas
                total_rows = 0
                total_cols = 0
                # Usar um iterador para evitar carregar o arquivo inteiro de uma vez
                csv_iterator = pd.read_csv(tmp_file_path, chunksize=1000, iterator=True)
                for i, chunk in enumerate(csv_iterator):
                    if i == 0:
                        total_cols = chunk.shape[1]
                    total_rows += chunk.shape[0]

                st.success(f"✅ Arquivo carregado com sucesso!")
                st.info(f"📊 **Dimensões:** {total_rows} linhas × {total_cols} colunas")
                
                # Mostrar preview dos Dados
                st.subheader("👀 Preview dos Dados")
                st.dataframe(df_preview, use_container_width=True)
                
                # Informações básicas sobre o dataset (usando as contagens de chunks)
                st.subheader("📈 Informações Básicas")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Linhas", total_rows)
                    st.metric("Colunas", total_cols)
                with col2:
                    # Para valores nulos e memória, precisaríamos carregar o DF completo ou usar uma abordagem mais complexa.
                    # Por simplicidade e para evitar estouro de memória, vamos omitir por enquanto ou usar uma estimativa.
                    st.metric("Valores Nulos (Estimativa)", "N/A") # Não é possível calcular sem carregar tudo
                    st.metric("Memória (Estimativa)", "N/A") # Não é possível calcular sem carregar tudo
                
                # Criar um banco de dados SQLite em disco (temporário) a partir do CSV
                db_path = os.path.join(tempfile.gettempdir(), "temp_db.db")
                from sqlalchemy import create_engine
                engine = create_engine(f"sqlite:///{db_path}")
                
                # Carregar o CSV para o SQLite em chunks para evitar estouro de memória
                chunksize = 1000  # Ajuste conforme necessário
                csv_iterator_to_sql = pd.read_csv(tmp_file_path, chunksize=chunksize, iterator=True)
                for i, chunk in enumerate(csv_iterator_to_sql):
                    chunk.to_sql("csv_data", engine, if_exists="append", index=False)
                
                # Criar o objeto SQLDatabase da LangChain com o engine
                db = SQLDatabase(engine=engine)
                st.session_state.db = db
                st.session_state.file_uploaded = True
                st.session_state.tmp_file_path = tmp_file_path # Guardar para limpeza
                
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
                        # Criar o agente LangChain SQL
                        llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name='llama-3.3-70b-versatile')
                        agent_executor = create_sql_agent(
                            llm=llm,
                            db=st.session_state.db,
                            agent_type="openai-tools",
                            verbose=False,
                            allow_dangerous_code=True,
                            agent_executor_kwargs={
                                "handle_parsing_errors": True
                            },
                            system_message="""
Você é um agente de IA especializado em analisar dados de CSVs. Sua tarefa é responder a perguntas sobre os dados da tabela \'csv_data\' em um banco de dados SQLite. 
Use as ferramentas disponíveis para inspecionar o esquema da tabela, gerar e executar consultas SQL para extrair as informações necessárias. 
Seja conciso e direto nas suas respostas. Se uma pergunta não puder ser respondida com os dados disponíveis, diga que não sabe.
"""
                        )

                        
                        # Executar a consulta
                        response = agent_executor.invoke({"input": prompt})
                        
                        # Exibir a resposta
                        st.markdown(response["output"])
                        
                        # Adicionar ao histórico
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response["output"]
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

