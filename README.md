# 🤖 Prática de Machine Learning com Scikit-Learn

Este repositório contém os exercícios práticos desenvolvidos para explorar conceitos fundamentais de **Machine Learning**, desde regressão linear simples até classificação binária, comparação de modelos de regressão e ajuste de hiperparâmetros com `GridSearchCV`.

---

## 📌 Conteúdo dos Exercícios

### 1. Parte (a) — Regressão Linear Simples
* **Objetivo:** Treinar um modelo de `LinearRegression` para aprender a relação entre dados numéricos simples (relação do triplo de um número).
* **Entrada:** Valores $X = [1, 2, 3, 4, 5]$ e respostas $y = [3, 6, 9, 12, 15]$.
* **Resultado:** O modelo previu corretamente **30.00** para a entrada $10$.

### 2. Parte (b) — Inspeção dos Parâmetros Aprendidos
* **Objetivo:** Explorar os parâmetros internos calculados pelo algoritmo através da equação da reta ($y = a \cdot X + b$).
* **Atributos Inspecionados:**
  * **Inclinação (`coef_`):** `[3.0]` (indica que $y$ aumenta 3 unidades para cada 1 unidade de $X$).
  * **Deslocamento (`intercept_`):** `0.0` (ponto de interceptação no eixo Y quando $X = 0$).

### 3. Parte (c) — Classificação Binária (Random Forest)
* **Objetivo:** Prever se vai chover (classe `1`) ou não (classe `0`) com base na **Temperatura** e **Umidade**.
* **Algoritmo:** `RandomForestClassifier(n_estimators=100, random_state=42)`.
* **Previsões Realizadas:**
  * `Temp: 26°C, Umidade: 65%` ➔ **Há previsão de chuva**
  * `Temp: 17°C, Umidade: 89%` ➔ **Não há previsão de chuva**

### 4. Parte (d) — Divisão Treino/Teste e Métricas
* **Objetivo:** Praticar a validação de modelos dividindo o conjunto de dados em treino (80%) e teste (20%) usando `train_test_split`.
* **Métrica Avaliada:** Acurácia no conjunto de teste via `modelo.score()`.
* **Conceito Chave:** Importância do `random_state` para reprodutibilidade dos dados em datasets de pequeno porte.

### 5. Parte (e) — Comparação de Modelos de Regressão
* **Objetivo:** Tratar variáveis categóricas com `pd.get_dummies()` a partir do dataset `vendas.csv` e comparar o desempenho de três regressores usando a métrica $R^2$ (`r2_score`):
  * **Linear Regression:** $R^2 \approx 0.99$ *(Melhor Modelo)*
  * **Decision Tree Regressor:** $R^2 \approx 0.91$
  * **Random Forest Regressor:** $R^2 \approx 0.94$

### 6. Bônus — Otimização de Hiperparâmetros (GridSearchCV)
* **Objetivo:** Ajustar e encontrar a melhor combinação de hiperparâmetros para modelos baseados em árvores.
* **Técnica:** `GridSearchCV` com validação cruzada (`cv=3`).
* **Hiperparâmetros Avaliados no Random Forest:**
  * `n_estimators`: `[10, 50, 100, 200]`
  * `max_depth`: `[None, 2, 4, 6]`
* **Resultado Encontrado:** `{'max_depth': None, 'n_estimators': 200}`

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

* **Python 3**
* **Pandas:** Leitura de arquivos CSV e manipulação de DataFrames/Encoding.
* **NumPy:** Manipulação e operações com arrays multidimensionais.
* **Matplotlib:** Visualização de dados e geração de gráficos.
* **Scikit-Learn:**
  * `LinearRegression`
  * `RandomForestClassifier` / `RandomForestRegressor`
  * `DecisionTreeRegressor`
  * `train_test_split`
  * `r2_score`
  * `GridSearchCV`

---

## 🚀 Como Executar

1. **Instale as dependências necessárias:**  
   Abra o terminal na pasta do projeto e instale todas as bibliotecas executando o comando: `pip install pandas numpy matplotlib scikit-learn`

2. **Execute o script principal:**  
   Para rodar todos os exercícios contidos no arquivo do projeto, execute no terminal o comando: `python exercicios.py`