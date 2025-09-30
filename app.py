
import streamlit as st
import pandas as pd
import os
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import matplotlib.pyplot as plt

def main():
    st.set_page_config(page_title="Agente de Análise de CSV", layout="wide")
    st.title("🤖 Agente de Análise de CSV")
    st.write("Faça upload de um arquivo CSV e faça perguntas sobre seus dados.")

    # Carregar a chave da API da OpenAI do ambiente
    try:
        openai_api_key = os.environ["OPENAI_API_KEY"]
    except KeyError:
        st.error("A variável de ambiente OPENAI_API_KEY não foi definida. Por favor, configure-a antes de executar.")
        return

    # Inicializar o histórico da conversa
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibir mensagens do histórico
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "response_plot" in message:
                st.pyplot(message["response_plot"])
            else:
                st.markdown(message["content"])

    # Componente de upload de arquivo
    uploaded_file = st.sidebar.file_uploader("Carregar arquivo CSV", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("Arquivo CSV carregado com sucesso!")
            st.sidebar.write("Amostra dos dados:")
            st.sidebar.dataframe(df.head())
            st.session_state.df = df
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar o arquivo: {e}")
            return

    # Entrada do usuário
    if prompt := st.chat_input("Qual sua pergunta sobre os dados?"): 
        if "df" not in st.session_state:
            st.warning("Por favor, carregue um arquivo CSV primeiro.")
            return

        # Adicionar mensagem do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gerar e exibir a resposta do agente
        with st.chat_message("assistant"):
            with st.spinner("Analisando os dados e gerando a resposta..."):
                try:
                    llm = OpenAI(temperature=0, api_key=openai_api_key)
                    agent = create_pandas_dataframe_agent(llm, st.session_state.df, verbose=True)
                    
                    # Intercepta a geração de gráficos
                    response = agent.run(prompt)

                    # Verifica se a resposta contém código para plotar
                    if "plt.show()" in response or "savefig" in response:
                        # Se o agente gerar código de plotagem, nós o executamos
                        fig = plt.figure()
                        exec_scope = {"df": st.session_state.df, "plt": plt}
                        exec(response, exec_scope)
                        st.pyplot(fig)
                        st.session_state.messages.append({"role": "assistant", "response_plot": fig})
                    else:
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_message = f"Ocorreu um erro: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()

