# 💳 Credit Card Fraud Detection Pipeline

Este repositório contém uma solução de Engenharia de Machine Learning corporativa para detecção de fraudes em transações de cartão de crédito (dados altamente desbalanceados com apenas 0,17% de fraudes). 

O projeto foi estruturado em camadas, oferecendo desde a execução automatizada via terminal até interfaces gráficas (Web e Desktop) para tomada de decisão em tempo real.

## 📁 Estrutura do Repositório

```text
├── core/
│   └── solucao_fraude.py     # Pipeline principal via CLI (Terminal)
├── src/
│   ├── web_app.py            # Dashboard interativo via Streamlit
│   └── desktop_app.py        # Aplicativo nativo em modo Dark (CustomTkinter)
├── assets/
│   └── *.png                 # Artefatos visuais e gráficos gerados
├── .gitignore                # Bloqueio de arquivos locais e venv
└── README.md                 # Documentação técnica do projeto
```

## 🚀 Como Executar as Versões

Certifique-se de estar com o ambiente virtual ativo (`source venv/bin/activate`).

### 1. Linha de Comando (CLI Core)
Para rodar o pipeline automatizado que treina os modelos, avalia as métricas de negócio (Recall/PR-AUC) e exporta a análise global:
```bash
python3 core/solucao_fraude.py
```

### 2. Dashboard Interativo Web (Streamlit)
Para abrir o painel dinâmico no navegador e tunar os limiares de decisão (*Threshold Tuning*) em tempo real:
```bash
streamlit run src/web_app.py
```

### 3. Aplicativo Nativo Desktop (CustomTkinter)
Para rodar a ferramenta standalone com interface gráfica escura e assíncrona diretamente no seu sistema operacional:
```bash
python3 src/desktop_app.py
```

## 🧠 Abordagem Sênior Aplicada
* **Métricas Corretas:** Substituição da métrica de Acurácia (ilusória para este problema) por foco estrito em **Recall** e **PR-AUC**.
* **Tratamento de Desbalanceamento:** Aplicação dinâmica de `scale_pos_weight` e `class_weight='balanced'` nativamente nos classificadores.
* **Interpretabilidade:** Integração de SHAP Values e gráficos de importância de recursos para justificar as decisões algorítmicas.
