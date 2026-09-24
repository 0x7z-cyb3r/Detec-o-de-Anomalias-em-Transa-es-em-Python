#!/usr/bin/env python3
import threading
import warnings
warnings.filterwarnings('ignore')

import tkinter as tk
import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc

import shap

# Configuração visual global do App (Modo Dark Sênior)
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class FraudApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("PROJETO DIO - Detector de Fraudes Sênior")
        self.geometry("1100x700")
        
        # Estado do Pipeline
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None]*4
        
        self.init_ui()
        self.start_load_data()

    def init_ui(self):
        # Layout principal em Grid (Esquerda: Controles | Direita: Gráficos/Métricas)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)
        
        # --- PAINEL LATERAL DE CONTROLES ---
        self.sidebar = ctk.CTkFrame(self, width=260, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.lbl_title = ctk.CTkLabel(self.sidebar, text="💳 FRAUD DETECTOR", font=ctk.CTkFont(size=20, weight="bold"))
        self.lbl_title.pack(pady=20, padx=10)
        
        # Status do Dataset
        self.lbl_status = ctk.CTkLabel(self.sidebar, text="Status: Carregando dados...", text_color="yellow")
        self.lbl_status.pack(pady=5)
        
        self.lbl_prop = ctk.CTkLabel(self.sidebar, text="Proporção: --", font=ctk.CTkFont(size=12))
        self.lbl_prop.pack(pady=5)
        
        # Seleção de Modelo
        self.lbl_model = ctk.CTkLabel(self.sidebar, text="Selecione o Modelo:", font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_model.pack(pady=(20, 5))
        
        self.cb_model = ctk.CTkComboBox(self.sidebar, values=["XGBoost (Advanced)", "Random Forest", "Logistic Regression"])
        self.cb_model.pack(pady=5, padx=10)
        
        # Slider de Threshold
        self.lbl_thresh = ctk.CTkLabel(self.sidebar, text="Limiar de Decisão (Threshold): 0.25", font=ctk.CTkFont(size=13))
        self.lbl_thresh.pack(pady=(20, 5))
        
        self.slider_thresh = ctk.CTkSlider(self.sidebar, from_=0.05, to=0.95, number_of_steps=18, command=self.update_thresh_label)
        self.slider_thresh.set(0.25)
        self.slider_thresh.pack(pady=5, padx=10)
        
        # Botão de Ação
        self.btn_run = ctk.CTkButton(self.sidebar, text="🚀 Treinar e Avaliar", command=self.start_training, state="disabled")
        self.btn_run.pack(pady=30, padx=10)
        
        # --- PAINEL CENTRAL DE RESULTADOS ---
        self.main_content = ctk.CTkFrame(self)
        self.main_content.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Dividindo o painel central em duas abas/linhas: Texto em cima, Plot embaixo
        self.txt_output = ctk.CTkTextbox(self.main_content, font=ctk.CTkFont(family="Courier", size=12))
        self.txt_output.pack(fill="both", expand=True, padx=10, pady=10)
        self.txt_output.insert("0.0", "Aguardando carregamento inicial para exibir o relatório de Machine Learning...")
        
        # Container para os gráficos do Matplotlib
        self.plot_frame = ctk.CTkFrame(self.main_content, height=300)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas = None

    def update_thresh_label(self, val):
        self.lbl_thresh.configure(text=f"Limiar de Decisão (Threshold): {val:.2f}")

    def start_load_data(self):
        # Carregamento assíncrono para a tela inicial abrir instantaneamente
        threading.Thread(target=self._async_load, daemon=True).start()

    def _async_load(self):
        url = "https://githubusercontent.com"
        try:
            # Tenta ler o arquivo local primeiro por segurança; se não achar, puxa da url simulada ou real
            try:
                self.df = pd.read_csv("creditcard.csv")
            except FileNotFoundError:
                self.df = pd.read_csv(url)
                
            # Pré-processamento Sênior básico
            self.df['Log_Amount'] = np.log1p(self.df['Amount'])
            scaler = StandardScaler()
            self.df['Scaled_Amount'] = scaler.fit_transform(self.df[['Amount']])
            self.df['Scaled_Time'] = scaler.fit_transform(self.df[['Time']])
            
            X = self.df.drop(columns=['Time', 'Amount', 'Class'])
            y = self.df['Class']
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y
            )
            
            fraudes = self.df['Class'].sum()
            total = len(self.df)
            prop = (fraudes / total) * 100
            
            self.lbl_status.configure(text="Status: Dataset Pronto!", text_color="green")
            self.lbl_prop.configure(text=f"Fraudes: {fraudes} ({prop:.3f}%)")
            self.btn_run.configure(state="normal")
        except Exception as e:
            self.lbl_status.configure(text="Status: Erro ao carregar dados", text_color="red")
            self.txt_output.delete("0.0", "end")
            self.txt_output.insert("0.0", f"Erro crítico de conectividade/leitura: {e}\nCertifique-se de colocar o arquivo 'creditcard.csv' na mesma pasta.")

    def start_training(self):
        self.btn_run.configure(state="disabled", text="Treinando...")
        threading.Thread(target=self._async_train, daemon=True).start()

    def _async_train(self):
        model_name = self.cb_model.get()
        thresh = self.slider_thresh.get()
        
        if model_name == "Logistic Regression":
            model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(class_weight='balanced', n_estimators=50, random_state=42, n_jobs=-1)
        else:
            scale_pos_weight_value = (len(self.y_train) - sum(self.y_train)) / sum(self.y_train)
            model = XGBClassifier(scale_pos_weight=scale_pos_weight_value, eval_metric='logloss', random_state=42)
            
        model.fit(self.X_train, self.y_train)
        
        y_probs = model.predict_proba(self.X_test)[:, 1]
        y_pred_custom = (y_probs >= thresh).astype(int)
        
        report_str = classification_report(self.y_test, y_pred_custom, target_names=['Normal', 'Fraude'])
        cm = confusion_matrix(self.y_test, y_pred_custom)
        
        # Atualizar a interface gráfica de forma segura
        self.after(0, lambda: self.update_ui_results(model_name, report_str, cm, model))

    def update_ui_results(self, model_name, report_str, cm, model):
        self.txt_output.delete("0.0", "end")
        self.txt_output.insert("0.0", f"=== RESULTADOS DO MODELO: {model_name} ===\n\n")
        self.txt_output.insert("end", report_str)
        self.txt_output.insert("end", f"\nMatriz de Confusão:\n{cm}\n")
        self.txt_output.insert("end", f"-> Fraudes Não Detectadas (Falsos Negativos): {cm[1][0]}\n")
        self.txt_output.insert("end", f"-> Alarmes Falsos (Falsos Positivos): {cm[0][1]}\n")
        
        # Limpar o gráfico antigo se houver
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            
        # Plotar Matriz de Confusão direto na Janela do App
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor('#2b2b2b')
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Normal', 'Fraude'], yticklabels=['Normal', 'Fraude'], ax=ax1)
        ax1.set_title("Matriz de Confusão Visual", color='white')
        ax1.set_ylabel('Realidade', color='white')
        ax1.set_xlabel('Predição', color='white')
        ax1.tick_params(colors='white')
        
        # Gráfico SHAP simplificado rápido em Matplotlib para não travar a janela
        ax2.set_facecolor('#2b2b2b')
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-10:]
            ax2.barh(range(len(indices)), importances[indices], color='#1f77b4', align='center')
            ax2.set_yticks(range(len(indices)))
            ax2.set_yticklabels([self.X_train.columns[i] for i in indices], color='white')
            ax2.set_title("Importância das Variáveis (Top 10)", color='white')
        else:
            ax2.text(0.5, 0.5, "SHAP/Importância indisponível\npara este modelo", color='white', ha='center', va='center')
        ax2.tick_params(colors='white')
        
        plt.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.btn_run.configure(state="normal", text="🚀 Treinar e Avaliar")

if __name__ == "__main__":
    app = FraudApp()
    app.mainloop()
