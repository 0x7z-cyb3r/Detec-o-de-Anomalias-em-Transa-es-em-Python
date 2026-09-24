#!/usr/bin/env python3
"""
Interface Gráfica Sênior - Pipeline de Detecção de Fraudes
Desafio de Projeto - DIO (Digital Innovation One)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc

import shap

# Configuração da página do Streamlit
st.set_page_config(page_title="Data Science Fraud Detection", page_icon="💳", layout="wide")

# Forçar estilo estético dos gráficos
sns.set_theme(style='whitegrid', palette='viridis')

@st.cache_data
def load_data(url_or_path):
    """Carrega os dados e armazena em cache para performance."""
    try:
        df = pd.read_csv(url_or_path)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None

def preprocess_data(df):
    """Pipeline de engenharia de dados e divisão estratificada."""
    df_proc = df.copy()
    df_proc['Log_Amount'] = np.log1p(df_proc['Amount'])
    
    scaler = StandardScaler()
    df_proc['Scaled_Amount'] = scaler.fit_transform(df_proc[['Amount']])
    df_proc['Scaled_Time'] = scaler.fit_transform(df_proc[['Time']])
    
    X = df_proc.drop(columns=['Time', 'Amount', 'Class'])
    y = df_proc['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

# --- INTERFACE VISUAL ---
st.title("💳 Monitor de Engenharia de Machine Learning: Detecção de Fraude")
st.markdown("---")

# Barra lateral para controle de dados
st.sidebar.header("📁 Gerenciamento de Dados")
data_source = st.sidebar.radio("Origem dos Dados:", ("Usar arquivo local (creditcard.csv)", "Baixar da Internet (GitHub)"))

if data_source == "Usar arquivo local (creditcard.csv)":
    DATA_PATH = "creditcard.csv"
else:
    DATA_PATH = "https://githubusercontent.com"

df = load_data(DATA_PATH)

if df is not None:
    # 1. Indicadores Globais
    st.header("📊 Análise de Desbalanceamento do Dataset")
    col1, col2, col3 = st.columns(3)
    
    total_transacoes = len(df)
    total_fraudes = df['Class'].sum()
    prop_fraude = (total_fraudes / total_transacoes) * 100
    
    col1.metric("Total de Transações", f"{total_transacoes:,}")
    col2.metric("Fraudes Detectadas", f"{total_fraudes:,}")
    col3.metric("Proporção de Fraude", f"{prop_fraude:.4f}%")
    
    # Executa o pré-processamento interno
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # 2. Configurações de Modelagem na Barra Lateral
    st.sidebar.header("⚙️ Hiperparâmetros do Modelo")
    selected_model_name = st.sidebar.selectbox("Escolha o Algoritmo:", ("XGBoost (Advanced)", "Random Forest", "Logistic Regression"))
    custom_threshold = st.sidebar.slider("Limiar de Decisão (Threshold):", 0.05, 0.95, 0.25, 0.05)
    
    # Inicialização do modelo escolhido
    if selected_model_name == "Logistic Regression":
        model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    elif selected_model_name == "Random Forest":
        model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42, n_jobs=-1)
    else:
        scale_pos_weight_value = (len(y_train) - sum(y_train)) / sum(y_train)
        model = XGBClassifier(scale_pos_weight=scale_pos_weight_value, eval_metric='logloss', random_state=42)
        
    # Botão para disparar o treinamento em tempo real
    if st.sidebar.button("🚀 Treinar e Avaliar Modelo"):
        with st.spinner(f"Treinando {selected_model_name}... Isso pode levar alguns segundos."):
            model.fit(X_train, y_train)
            
            # Predições baseadas no limiar customizado
            y_probs = model.predict_proba(X_test)[:, 1]
            y_pred_custom = (y_probs >= custom_threshold).astype(int)
            
            st.markdown("---")
            st.header(f"📈 Resultados de Performance: {selected_model_name}")
            
            # Métricas em formato texto estruturado
            col_metrics, col_graph = st.columns(2)
            
            with col_metrics:
                st.subheader("📋 Relatório de Classificação")
                report = classification_report(y_test, y_pred_custom, target_names=['Normal', 'Fraude'], output_dict=True)
                
                # Exibe métricas de Fraude em destaque
                st.write(pd.DataFrame(report).transpose())
                
                cm = confusion_matrix(y_test, y_pred_custom)
                st.error(f"🚨 Falsos Negativos (Fraudes Perdidas): {cm[1][0]}")
                st.warning(f"⚠️ Falsos Positivos (Alarmes Falsos): {cm[0][1]}")
                
            with col_graph:
                st.subheader("🎯 Matriz de Confusão Visual")
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fraude'], yticklabels=['Normal', 'Fraude'], ax=ax_cm)
                plt.ylabel('Realidade')
                plt.xlabel('Predição do Modelo')
                st.pyplot(fig_cm)
                
            # 3. Explicação com SHAP
            st.markdown("---")
            st.header("🧠 Interpretabilidade com SHAP (Importância Global)")
            with st.spinner("Calculando SHAP values..."):
                X_shap_sample = X_test.sample(min(100, len(X_test)), random_state=42)
                
                # SHAP seguro para os diferentes tipos de classificadores
                if selected_model_name == "Logistic Regression":
                    explainer = shap.LinearExplainer(model, X_train)
                    shap_values = explainer.shap_values(X_shap_sample)
                else:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_shap_sample, check_additivity=False)
                
                fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
                shap.summary_plot(shap_values, X_shap_sample, plot_type="bar", show=False)
                st.pyplot(fig_shap)
else:
    st.warning("Aguardando carregamento de dados válido.")
