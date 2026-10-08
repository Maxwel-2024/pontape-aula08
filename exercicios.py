import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Montar o DataFrame com os dados da tabela
dados = {
    "Temperatura": [30, 25, 20, 15, 32, 28, 22, 18],
    "Umidade": [60, 70, 80, 90, 55, 75, 88, 92],
    "Choveu": [1, 1, 0, 0, 1, 1, 0, 0],
}
df = pd.DataFrame(dados)

# 2. Separar as variáveis de entrada (X) e a meta (y)
X = df[["Temperatura", "Umidade"]]
y = df["Choveu"]

# 3. Treinar o RandomForestClassifier
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X, y)

# 4. Novos dados para previsão
novos_dados = [[26, 65], [17, 89]]

# 5. Realizar as previsões
previsoes = modelo.predict(novos_dados)

# 6. Exibir o resultado formatado
for dados_entrada, prev in zip(novos_dados, previsoes):
    texto = (
        "Há previsão de chuva" if prev == 1 else "Não há previsão de chuva"
    )
    print(
        f"Temperatura = {dados_entrada[0]} , Umidade = {dados_entrada[1]}: {texto}"
    )