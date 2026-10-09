import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

# 1. Carregar os dados e aplicar get_dummies
df = pd.read_csv("vendas.csv")
df_encoded = pd.get_dummies(df, columns=["Promocao"], drop_first=True)

# 2. Separar X e y
X = df_encoded.drop(columns=["Vendas"])
y = df_encoded["Vendas"]

# 3. Dividir em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Definir a grade de parâmetros para testar
param_grid = {"n_estimators": [10, 50, 100, 200], "max_depth": [None, 2, 4, 6]}

# 5. Instanciar e rodar o GridSearchCV no RandomForestRegressor
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=rf, param_grid=param_grid, cv=3, scoring="r2"
)
grid_search.fit(X_treino, y_treino)

# 6. Exibir os melhores parâmetros encontrados
print(f"Melhores parâmetros: {grid_search.best_params_}")