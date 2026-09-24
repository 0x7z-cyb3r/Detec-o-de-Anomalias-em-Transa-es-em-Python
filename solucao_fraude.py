#!/usr/bin/env python3
"""
Pipeline de Machine Learning Sênior - Detecção de Fraude em Cartão de Crédito
Desafio de Projeto - DIO (Digital Innovation One)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc

import shap

# Configuração visual dos plots
sns.set_theme(style='whitegrid', palette='viridis')

def load_data(url):
    """Carrega o dataset e faz a validação inicial do desbalanceamento."""
    print("[INFO] Carregando o dataset de transações de cartão de crédito...")
    df = pd.read_csv(url)
    
    print(f"Dimensões do dataset: {df.shape}")
    print("\nProporção das classes (%):")
    print(df['Class'].value_counts(normalize=True) * 100)
    print(f"\nValores ausentes no dataset: {df.isnull().sum().sum()}")
    return df

def preprocess_data(df):
    """Pipeline de engenharia de dados e divisão estratificada."""
    print("\n[INFO] Iniciando pipeline de pré-processamento...")
    
    # Engenharia de Recursos: Aplicação de escala logarítmica para mitigar skewness de 'Amount'
    df['Log_Amount'] = np.log1p(df['Amount'])
    
    # Padronização das variáveis de tempo e valor bruto
    scaler = StandardScaler()
    df['Scaled_Amount'] = scaler.fit_transform(df[['Amount']])
    df['Scaled_Time'] = scaler.fit_transform(df[['Time']])
    
    # Definição de Features (X) e Target (y) desconsiderando os dados brutos antigos
    X = df.drop(columns=['Time', 'Amount', 'Class'])
    y = df['Class']
    
    # CORREÇÃO: Alterado de 'test_split' para 'test_size'
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Proporção de Fraude no Treino: {y_train.mean():.4%}")
    print(f"Proporção de Fraude no Teste: {y_test.mean():.4%}")
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """Instancia e treina os modelos utilizando tratamento de pesos para classes imbalanced."""
    print("\n[INFO] Iniciando treinamento dos modelos...")
    
    # Cálculo dinâmico do fator de custo para o XGBoost
    scale_pos_weight_value = (len(y_train) - sum(y_train)) / sum(y_train)
    
    models = {
        "Logistic Regression (Baseline)": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Random Forest (Robust)": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost (Advanced)": XGBClassifier(scale_pos_weight=scale_pos_weight_value, eval_metric='logloss', random_state=42)
    }
    
    for name, model in models.items():
        print(f"-> Ajustando modelo: {name}...")
        model.fit(X_train, y_train)
        
    return models

def evaluate_models(models, X_test, y_test):
    """Avaliação rigorosa dos modelos focando em Precision-Recall."""
    print("\n[INFO] Iniciando avaliação analítica...")
    for name, model in models.items():
        print("="*60)
        print(f" MODELO: {name} ")
        print("="*60)
        
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]
        
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraude']))
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"Matriz de Confusão:\n{cm}")
        print(f"-> Falsos Negativos (Fraudes perdidas): {cm[1][0]}")
        print(f"-> Falsos Positivos (Alarmes falsos): {cm[0][1]}\n")
        
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(y_test, y_probs)
        print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC (Métrica Crítica): {pr_auc:.4f}\n")

def optimize_threshold(model, X_test, y_test, threshold=0.25):
    """Ajuste do Limiar de Decisão para ganho de sensibilidade/Recall."""
    print("="*60)
    print(f"[OTIMIZAÇÃO] Avaliação com Threshold customizado em: {threshold}")
    print("="*60)
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred_custom = (y_probs >= threshold).astype(int)
    print(classification_report(y_test, y_pred_custom, target_names=['Normal', 'Fraude']))

def explain_model(model, X_test):
    """Interpretabilidade global utilizando SHAP valores em amostragem."""
    print("\n[INFO] Calculando SHAP values (Amostragem otimizada para o terminal do Kali)...")
    X_shap_sample = X_test.sample(100, random_state=42)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap_sample, check_additivity=False)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_shap_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('shap_summary_importance.png')
    print("[SUCESSO] Gráfico de importância SHAP salvo como 'shap_summary_importance.png'")

if __name__ == "__main__":
    DATA_URL = "creditcard.csv"
    
    try:
        df = load_data(DATA_URL)
        X_train, X_test, y_train, y_test = preprocess_data(df)
        models = train_models(X_train, y_train)
        evaluate_models(models, X_test, y_test)
        
        champion_model = models["XGBoost (Advanced)"]
        optimize_threshold(champion_model, X_test, y_test, threshold=0.25)
        explain_model(champion_model, X_test)
        
    except KeyError:
        print("[ERRO] A coluna 'Class' não foi encontrada. Verifique se o dataset está estruturado corretamente.")
