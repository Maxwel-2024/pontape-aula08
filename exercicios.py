import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# 1. Carregar o arquivo vendas.csv
df = pd.read_csv("vendas.csv")

# 2. Aplicar get_dummies na coluna categórica 'Promocao'
df_encoded = pd.get_dummies(df, columns=["Promocao"], drop_first=True)

# 3. Separar X e y
X = df_encoded.drop(columns=["Vendas"])
y = df_encoded["Vendas"]

# 4. Dividir treino/teste (test_size=0.2, random_state=42)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Instanciar e treinar os 3 modelos de regressão
modelos = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
}

r2_resultados = {}

for nome, modelo in modelos.items():
    modelo.fit(X_treino, y_treino)
    previsoes = modelo.predict(X_teste)
    r2_resultados[nome] = r2_score(y_teste, previsoes)

# 6. Identificar o modelo com o maior R²
melhor_modelo = max(r2_resultados, key=r2_resultados.get)

# Exibir os resultados formatados
print(f"R² Linear:        {r2_resultados['Linear Regression']:.2f}")
print(f"R² Árvore:        {r2_resultados['Decision Tree']:.2f}")
print(f"R² Random Forest: {r2_resultados['Random Forest']:.2f}")
print(f"Melhor modelo:    {melhor_modelo}")