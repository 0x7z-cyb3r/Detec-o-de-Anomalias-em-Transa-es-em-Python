# Detecção de Fraude em Cartão de Crédito com Machine Learning

### 1. O Problema e a Acurácia
Em cenários onde apenas 0,17% das transações são fraudes, a acurácia é uma métrica ilusória. Um modelo que chute "normal" para tudo teria 99,83% de acurácia, mas não detectaria nenhuma fraude. Focamos em **Recall** para capturar o máximo de fraudes possíveis e **PR-AUC** para equilibrar a experiência do cliente (evitar falsos bloqueios).

### 2. Preparação de Dados
* Variáveis `Time` e `Amount` padronizadas com `StandardScaler`.
* Divisão de treino/teste com estratificação (`stratify=y`) para manter os 0,17% de fraudes em ambas as bases.

### 3. Modelos Comparados
* **Regressão Logística**: Baseline com alto recall, mas muitos alarmes falsos.
* **Random Forest**: Robusto, boa capacidade de generalização.
* **XGBoost**: Melhor performance geral medida pela curva Precisão-Recall (PR-AUC).

### 4. Tomada de Decisão e SHAP
* Reduzimos o limiar de decisão (`threshold`) padrão de 0,5 para 0,25 para aumentar a sensibilidade a fraudes.
* O **SHAP** revelou que os componentes latentes (derivados de PCA) foram os fatores de maior peso para a classificação do modelo.
