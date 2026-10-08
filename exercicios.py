from sklearn.linear_model import LinearRegression

# 1. Dados de entrada (X) e saídas esperadas (y)
# Lembre-se: X precisa ser uma matriz 2D (lista de listas ou array 2D)
X = [[1], [2], [3], [4], [5]]
y = [3, 6, 9, 12, 15]

# 2. Instanciar o modelo de Regressão Linear
modelo = LinearRegression()

# 3. Treinar o modelo com os dados de exemplo
modelo.fit(X, y)

# 4. Fazer a previsão para o número 10
previsao = modelo.predict([[10]])

# 5. Exibir o resultado formatado
print(f"Previsão para 10: {previsao[0]:.2f}")