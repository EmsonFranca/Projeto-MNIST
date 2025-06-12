# Parte principal do projeto responsável por:
# iniciar o treinamento, carregar os dados e montar as estruturas da rede.

# Explicação dos imports:
# avaliacao        → traz funções como cálculo de acurácia e exibição da matriz de confusão.
# data_loader      → função que realiza o carregamento dos dados do MNIST.
# MLP              → classe do Perceptron Multicamadas utilizada para definir e treinar a rede.
# matplotlib.pyplot→ biblioteca gráfica usada para visualizar os resultados.
# visualizacao     → funções que exibem comparações entre os dígitos reais e os previstos.

from avaliacao import calcular_acuracia, matriz_confusao, exibir_matriz_confusao
from data_loader import carregar_dados_mnist
from visualizacao import mostrar_previsoes
from mlp import MLP
import matplotlib.pyplot as plt

# Carrega os dados do conjunto MNIST (imagens e rótulos de treino e teste)
X_train, y_train, X_test, y_test = carregar_dados_mnist()

# Inicializa a rede neural com:
# - 128 neurônios na camada oculta
# - taxa de aprendizado de 0.1
# Treina a rede por 100 épocas e armazena o histórico da função de perda
modelo = MLP(hidden_size=128, taxa_aprendizado=0.1)
historico = modelo.treinar(X_train, y_train, 100)

# Realiza as previsões com os dados de teste
# e calcula a acurácia final do modelo
pred_test = modelo.forward(X_test)
acc = calcular_acuracia(pred_test, y_test)
print(f"Acurácia no teste: {acc:.8f}")

# Gera a matriz de confusão comparando os rótulos reais com os preditos
# e exibe a matriz de forma visual
y_predito = pred_test.argmax(axis=1)
matriz = matriz_confusao(y_test, y_predito)
exibir_matriz_confusao(matriz)

# Mostra graficamente algumas imagens de teste com seus respectivos rótulos
# previstos pela rede para análise visual da performance
mostrar_previsoes(X_test, y_test, y_predito, quantidade=25)

# Plota o gráfico da função de perda ao longo das épocas
# para visualização do aprendizado da rede
plt.plot(historico)
plt.title("Função de Perda por Época")
plt.xlabel("Época")
plt.ylabel("Perda")
plt.grid(True)
plt.show()
