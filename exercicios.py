import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Dataset da parte (c)
dados = {
    "Temperatura": [30, 25, 20, 15, 32, 28, 22, 18],
    "Umidade": [60, 70, 80, 90, 55, 75, 88, 92],
    "Choveu": [1, 1, 0, 0, 1, 1, 0, 0],
}
df = pd.DataFrame(dados)

X = df[["Temperatura", "Umidade"]]
y = df["Choveu"]

# 2. Dividir em treino (80%) e teste (20%)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Treinar o modelo apenas com o conjunto de treino
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_treino, y_treino)

# 4. Avaliar a acurácia no conjunto de teste
acuracia = modelo.score(X_teste, y_teste)

# 5. Exibir os resultados
print(f"Tamanho do treino: {len(X_treino)}")
print(f"Tamanho do teste: {len(X_teste)}")
print(f"Acurácia no teste: {acuracia:.2f}")