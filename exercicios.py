from sklearn.linear_model import LinearRegression

# 1. Dados de treino (X em 2D e y em 1D)
X = [[1], [2], [3], [4], [5]]
y = [3, 6, 9, 12, 15]

# 2. Instanciar e treinar o modelo
modelo = LinearRegression()
modelo.fit(X, y)

# 3. Extrair os parâmetros aprendidos (tratando o -0.0 com abs)
inclinacao = modelo.coef_
deslocamento = abs(modelo.intercept_)

# 4. Exibir apenas a saída da Parte (b)
print(f"Inclinação (coef_): {inclinacao}")
print(f"Deslocamento (intercept_): {deslocamento:.1f}")