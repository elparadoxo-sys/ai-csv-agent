import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import json
from datetime import datetime
import sys

# Verificar e instalar dependências
def check_and_install_dependencies():
    required_packages = {
        'plotly': 'plotly',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'scipy': 'scipy',
        'scikit-learn': 'sklearn',
        'langchain': 'langchain',
        'langchain-groq': 'langchain_groq',
        'python-dotenv': 'dotenv'
    }
    
    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name.split('.')[0])
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        st.error(f"⚠️ Pacotes faltando: {', '.join(missing_packages)}")
        st.info("Execute: `pip install " + " ".join(missing_packages) + "`")
        st.stop()

check_and_install_dependencies()

# Importações principais
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import io
import base64

# Carregar variáveis de ambiente
load_dotenv()

# Configurar estilo dos gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

class CSVAnalyzer:
    """Classe para análise de dados CSV com memória de conclusões"""
    
    def __init__(self, df):
        self.df = df
        self.conclusions = []
        self.analyses_performed = []
        
    def add_conclusion(self, conclusion_type, conclusion_text):
        """Adiciona uma conclusão à memória"""
        self.conclusions.append({
            'timestamp': datetime.now().isoformat(),
            'type': conclusion_type,
            'text': conclusion_text
        })
        
    def add_analysis(self, analysis_type, details):
        """Registra uma análise realizada"""
        self.analyses_performed.append({
            'timestamp': datetime.now().isoformat(),
            'type': analysis_type,
            'details': details
        })
        
    def get_conclusions_summary(self):
        """Retorna resumo de todas as conclusões"""
        if not self.conclusions:
            return "Nenhuma análise foi realizada ainda."
        
        summary = "=== CONCLUSÕES CONSOLIDADAS ===\n\n"
        for i, conc in enumerate(self.conclusions, 1):
            summary += f"{i}. [{conc['type']}] {conc['text']}\n\n"
        return summary
    
    def get_basic_info(self):
        """Informações básicas do dataset"""
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        
        conclusion = f"Dataset com {info['shape'][0]} linhas e {info['shape'][1]} colunas. "
        if sum(info['missing_values'].values()) > 0:
            conclusion += f"Contém valores ausentes em {sum(1 for v in info['missing_values'].values() if v > 0)} colunas."
        else:
            conclusion += "Não há valores ausentes."
        
        self.add_conclusion("Informações Básicas", conclusion)
        self.add_analysis("basic_info", info)
        
        return json.dumps(info, indent=2, default=str)
    
    def get_descriptive_stats(self):
        """Estatísticas descritivas"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        stats_dict = {}
        
        for col in numeric_cols:
            stats_dict[col] = {
                'mean': float(self.df[col].mean()),
                'median': float(self.df[col].median()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'q25': float(self.df[col].quantile(0.25)),
                'q75': float(self.df[col].quantile(0.75))
            }
        
        conclusion = f"Analisadas {len(numeric_cols)} variáveis numéricas. "
        high_var_cols = [col for col, s in stats_dict.items() if s['std'] > s['mean']]
        if high_var_cols:
            conclusion += f"Colunas com alta variabilidade: {', '.join(high_var_cols[:3])}."
        
        self.add_conclusion("Estatísticas Descritivas", conclusion)
        self.add_analysis("descriptive_stats", stats_dict)
        
        return json.dumps(stats_dict, indent=2)
    
    def detect_outliers(self, column=None):
        """Detecta outliers usando IQR"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if column and column in numeric_cols:
            cols_to_check = [column]
        else:
            cols_to_check = numeric_cols
        
        outliers_info = {}
        for col in cols_to_check:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outliers_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(self.df)) * 100,
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
        
        cols_with_outliers = [col for col, info in outliers_info.items() if info['count'] > 0]
        if cols_with_outliers:
            conclusion = f"Detectados outliers em {len(cols_with_outliers)} colunas. "
            top_outlier_col = max(outliers_info.items(), key=lambda x: x[1]['percentage'])
            conclusion += f"Coluna com mais outliers: {top_outlier_col[0]} ({top_outlier_col[1]['percentage']:.2f}%)."
        else:
            conclusion = "Nenhum outlier detectado nas colunas numéricas."
        
        self.add_conclusion("Detecção de Outliers", conclusion)
        self.add_analysis("outliers", outliers_info)
        
        return json.dumps(outliers_info, indent=2)
    
    def calculate_correlation(self):
        """Calcula matriz de correlação"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return "Não há colunas numéricas suficientes para correlação."
        
        corr_matrix = self.df[numeric_cols].corr()
        
        # Encontrar correlações mais fortes
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'col1': corr_matrix.columns[i],
                    'col2': corr_matrix.columns[j],
                    'correlation': float(corr_matrix.iloc[i, j])
                })
        
        corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        top_corr = corr_pairs[:5]
        
        conclusion = "Análise de correlação concluída. "
        if top_corr and abs(top_corr[0]['correlation']) > 0.7:
            conclusion += f"Correlação forte encontrada entre {top_corr[0]['col1']} e {top_corr[0]['col2']} ({top_corr[0]['correlation']:.3f})."
        else:
            conclusion += "Não foram encontradas correlações fortes entre as variáveis."
        
        self.add_conclusion("Análise de Correlação", conclusion)
        self.add_analysis("correlation", {'top_correlations': top_corr})
        
        return json.dumps({'top_correlations': top_corr}, indent=2)
    
    def perform_clustering(self, n_clusters=3, method='kmeans'):
        """Realiza análise de clusters"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return "Não há colunas numéricas suficientes para clustering."
        
        # Preparar dados
        X = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        
        clusters = clusterer.fit_predict(X_scaled)
        unique_clusters = len(set(clusters))
        
        cluster_info = {
            'method': method,
            'n_clusters': unique_clusters,
            'cluster_sizes': {int(k): int(v) for k, v in pd.Series(clusters).value_counts().to_dict().items()}
        }
        
        conclusion = f"Identificados {unique_clusters} clusters/agrupamentos nos dados usando {method}. "
        sizes = list(cluster_info['cluster_sizes'].values())
        if max(sizes) > len(self.df) * 0.8:
            conclusion += "Um cluster dominante sugere dados homogêneos."
        else:
            conclusion += "Clusters balanceados indicam grupos distintos nos dados."
        
        self.add_conclusion("Análise de Clusters", conclusion)
        self.add_analysis("clustering", cluster_info)
        
        return json.dumps(cluster_info, indent=2)

class GraphGenerator:
    """Classe para geração de gráficos"""
    
    def __init__(self, df):
        self.df = df
    
    def create_histogram(self, column, bins=30):
        """Cria histograma"""
        fig = px.histogram(self.df, x=column, nbins=bins, 
                          title=f'Distribuição de {column}',
                          labels={column: column, 'count': 'Frequência'})
        fig.update_layout(showlegend=False)
        return fig
    
    def create_boxplot(self, columns):
        """Cria boxplot para detectar outliers"""
        if isinstance(columns, str):
            columns = [columns]
        
        fig = go.Figure()
        for col in columns:
            fig.add_trace(go.Box(y=self.df[col], name=col))
        
        fig.update_layout(title='Boxplot - Detecção de Outliers',
                         yaxis_title='Valor')
        return fig
    
    def create_correlation_heatmap(self):
        """Cria heatmap de correlação"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto='.2f',
                       title='Matriz de Correlação',
                       color_continuous_scale='RdBu_r',
                       aspect='auto')
        return fig
    
    def create_scatter_plot(self, x_col, y_col, color_col=None):
        """Cria gráfico de dispersão"""
        fig = px.scatter(self.df, x=x_col, y=y_col, color=color_col,
                        title=f'{y_col} vs {x_col}',
                        labels={x_col: x_col, y_col: y_col})
        return fig
    
    def create_time_series(self, time_col, value_col):
        """Cria gráfico de série temporal"""
        fig = px.line(self.df, x=time_col, y=value_col,
                     title=f'{value_col} ao longo do tempo',
                     labels={time_col: 'Tempo', value_col: value_col})
        return fig
    
    def create_distribution_comparison(self, column, group_by):
        """Compara distribuições entre grupos"""
        fig = px.box(self.df, x=group_by, y=column,
                    title=f'Distribuição de {column} por {group_by}',
                    labels={group_by: group_by, column: column})
        return fig

def main():
    st.set_page_config(
        page_title="🤖 Agente EDA Completo",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Agente de Análise Exploratória de Dados (EDA)")
    st.markdown("""
    Sistema completo de análise de dados com **IA, Geração de Gráficos e Memória de Conclusões**.
    """)

    # Verificar API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("⚠️ Configure a variável de ambiente GROQ_API_KEY")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("📁 Upload de Arquivo")
        uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Arquivo carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")
                
                # Inicializar analisador
                if 'analyzer' not in st.session_state:
                    st.session_state.analyzer = CSVAnalyzer(df)
                    st.session_state.graph_gen = GraphGenerator(df)
                    st.session_state.df = df
                
                # Preview
                with st.expander("👀 Preview dos Dados"):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Info básica
                with st.expander("📊 Informações do Dataset"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Linhas", df.shape[0])
                        st.metric("Colunas", df.shape[1])
                    with col2:
                        st.metric("Valores Nulos", df.isnull().sum().sum())
                        st.metric("Memória", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                    
                    st.write("**Tipos de Dados:**")
                    st.write(df.dtypes.value_counts())
                
            except Exception as e:
                st.error(f"❌ Erro ao carregar arquivo: {str(e)}")

    # Área principal
    if 'analyzer' in st.session_state:
        
        # Criar ferramentas para o agente
        def tool_get_info(query: str) -> str:
            """Obtém informações básicas do dataset"""
            return st.session_state.analyzer.get_basic_info()
        
        def tool_statistics(query: str) -> str:
            """Calcula estatísticas descritivas"""
            return st.session_state.analyzer.get_descriptive_stats()
        
        def tool_detect_outliers(query: str) -> str:
            """Detecta outliers nos dados"""
            return st.session_state.analyzer.detect_outliers()
        
        def tool_correlation(query: str) -> str:
            """Calcula correlações entre variáveis"""
            return st.session_state.analyzer.calculate_correlation()
        
        def tool_clustering(query: str) -> str:
            """Realiza análise de clusters"""
            return st.session_state.analyzer.perform_clustering()
        
        def tool_conclusions(query: str) -> str:
            """Retorna todas as conclusões obtidas nas análises"""
            return st.session_state.analyzer.get_conclusions_summary()
        
        def tool_create_histogram(column: str) -> str:
            """Cria histograma de uma coluna"""
            try:
                fig = st.session_state.graph_gen.create_histogram(column)
                st.session_state.current_plot = fig
                return f"Gráfico de histograma criado para a coluna {column}"
            except Exception as e:
                return f"Erro ao criar gráfico: {str(e)}"
        
        def tool_create_boxplot(columns: str) -> str:
            """Cria boxplot para detectar outliers"""
            try:
                cols = [c.strip() for c in columns.split(',')]
                fig = st.session_state.graph_gen.create_boxplot(cols)
                st.session_state.current_plot = fig
                return f"Gráfico boxplot criado para: {columns}"
            except Exception as e:
                return f"Erro ao criar gráfico: {str(e)}"
        
        def tool_create_heatmap(query: str) -> str:
            """Cria heatmap de correlação"""
            try:
                fig = st.session_state.graph_gen.create_correlation_heatmap()
                st.session_state.current_plot = fig
                return "Heatmap de correlação criado"
            except Exception as e:
                return f"Erro ao criar gráfico: {str(e)}"
        
        def tool_create_scatter(columns: str) -> str:
            """Cria gráfico de dispersão entre duas colunas (formato: col1,col2)"""
            try:
                cols = [c.strip() for c in columns.split(',')]
                if len(cols) >= 2:
                    fig = st.session_state.graph_gen.create_scatter_plot(cols[0], cols[1])
                    st.session_state.current_plot = fig
                    return f"Gráfico de dispersão criado: {cols[0]} vs {cols[1]}"
                return "Forneça duas colunas separadas por vírgula"
            except Exception as e:
                return f"Erro ao criar gráfico: {str(e)}"
        
        tools = [
            Tool(name="informacoes_basicas", func=tool_get_info, 
                 description="Obtém informações básicas do dataset: dimensões, tipos de dados, valores ausentes"),
            Tool(name="estatisticas_descritivas", func=tool_statistics,
                 description="Calcula estatísticas descritivas: média, mediana, desvio padrão, min, max, quartis"),
            Tool(name="detectar_outliers", func=tool_detect_outliers,
                 description="Detecta valores atípicos (outliers) usando método IQR"),
            Tool(name="calcular_correlacao", func=tool_correlation,
                 description="Calcula matriz de correlação entre variáveis numéricas"),
            Tool(name="analise_clusters", func=tool_clustering,
                 description="Realiza análise de agrupamentos (clusters) nos dados"),
            Tool(name="obter_conclusoes", func=tool_conclusions,
                 description="Retorna TODAS as conclusões e insights obtidos nas análises realizadas"),
            Tool(name="criar_histograma", func=tool_create_histogram,
                 description="Cria histograma de distribuição. Input: nome da coluna"),
            Tool(name="criar_boxplot", func=tool_create_boxplot,
                 description="Cria boxplot para detectar outliers. Input: nome(s) da(s) coluna(s)"),
            Tool(name="criar_heatmap_correlacao", func=tool_create_heatmap,
                 description="Cria heatmap da matriz de correlação"),
            Tool(name="criar_grafico_dispersao", func=tool_create_scatter,
                 description="Cria gráfico de dispersão. Input: 'coluna_x,coluna_y'"),
        ]
        
        # Inicializar memória e agente
        if 'memory' not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Template do agente
        columns_list = ", ".join(st.session_state.df.columns)
        
        template = """Você é um agente especialista em Análise Exploratória de Dados (EDA).

Colunas disponíveis no dataset: """ + columns_list + """

Você tem acesso às seguintes ferramentas:
{tools}

Use o seguinte formato:

Question: a pergunta ou tarefa que você deve responder
Thought: você deve sempre pensar sobre o que fazer
Action: a ação a tomar, deve ser uma de [{tool_names}]
Action Input: o input para a ação
Observation: o resultado da ação
... (este Thought/Action/Action Input/Observation pode repetir N vezes)
Thought: Agora sei a resposta final
Final Answer: a resposta final para a pergunta original

INSTRUÇÕES IMPORTANTES:
1. Para análises estatísticas, use as ferramentas apropriadas
2. Para criar gráficos, use as ferramentas de visualização
3. SEMPRE que realizar uma análise, interprete os resultados e tire conclusões
4. Quando perguntado sobre conclusões, use a ferramenta 'obter_conclusoes'
5. Seja claro, objetivo e forneça insights acionáveis

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        )
        
        # Criar agente
        llm = ChatGroq(
            temperature=0,
            groq_api_key=groq_api_key,
            model_name='llama-3.3-70b-versatile'
        )
        
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            return_intermediate_steps=False
        )
        
        # Interface de chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "plot" in message and message["plot"]:
                    st.plotly_chart(message["plot"], use_container_width=True)
        
        if prompt_input := st.chat_input("Faça uma pergunta sobre os dados..."):
            st.session_state.messages.append({"role": "user", "content": prompt_input})
            
            with st.chat_message("user"):
                st.markdown(prompt_input)
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analisando..."):
                    try:
                        st.session_state.current_plot = None
                        
                        # Invocar agente sem parâmetros extras
                        response = agent_executor.invoke({
                            "input": prompt_input
                        })
                        
                        st.markdown(response["output"])
                        
                        plot_to_save = None
                        if st.session_state.current_plot is not None:
                            st.plotly_chart(st.session_state.current_plot, use_container_width=True)
                            plot_to_save = st.session_state.current_plot
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response["output"],
                            "plot": plot_to_save
                        })
                        
                    except Exception as e:
                        error_msg = f"❌ Erro: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "plot": None
                        })
        
        # Exemplos
        with st.expander("💡 Exemplos de Perguntas"):
            st.markdown("""
            **Análise Descritiva:**
            - Quais são as estatísticas descritivas dos dados?
            - Mostre um histograma da coluna Amount
            - Crie um boxplot para detectar outliers
            
            **Análise de Padrões:**
            - Existem correlações fortes entre as variáveis?
            - Mostre um heatmap de correlação
            - Existem clusters nos dados?
            
            **Conclusões:**
            - Quais conclusões você obteve dos dados?
            - Resuma todos os insights encontrados
            - O que você aprendeu sobre este dataset?
            """)
        
        # Botão para ver conclusões
        if st.sidebar.button("📋 Ver Todas as Conclusões"):
            with st.sidebar:
                st.markdown("### Conclusões Consolidadas")
                st.text(st.session_state.analyzer.get_conclusions_summary())
    
    else:
        st.info("👆 Faça upload de um arquivo CSV para começar!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("🧠 IA Avançada")
            st.write("Agente ReAct com memória e múltiplas ferramentas")
        with col2:
            st.subheader("📊 Gráficos Interativos")
            st.write("Histogramas, boxplots, heatmaps e scatter plots")
        with col3:
            st.subheader("💭 Memória de Conclusões")
            st.write("Armazena e resume insights obtidos")

if __name__ == "__main__":
    main()
