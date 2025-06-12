import numpy as np
from ativacao import relu, relu_derivada, softmax
from perdas import cross_entropy


class MLP:
    # Essa é a rede neural em MPL! Iniciada com as entradas, neurônios,
    # classes, taxa de aprendizado
    def __init__(self, input_size=784, hidden_size=64, output_size=10,
                taxa_aprendizado=0.01):
        self.lr = taxa_aprendizado
        # Matrizes de pesos entre camadas, começam aleatórios
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        # Bias da primeira camada, começa com zeros
        # A matriz de pesos da segunda camada, que vai da camada oculta para a saída
        self.w2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        # A multiplicação dos pesos iniciais por 0.01 é uma técnica essencial para manter
        # os valores pequenos no início do treinamento. Isso ajuda a evitar a saturação
        # dos neurônios, o que poderia dificultar a propagação dos sinais e prejudicar
        # o aprendizado da rede neural.

    def forward(self, X):
        # Essa é a propragação para frente!
        # Resultado da multiplicação dos dados pelos pesoss + bias
        self.z1 = X @ self.w1 + self.b1 # O operador @ é uma multiplicação de matrizes em Python (numpy)
        # Aplicando ReLU
        self.a1 = relu(self.z1)
        # Entrada bruta da camada de saída
        self.z2 = self.a1 @ self.w2 + self.b2
        # Aplicando softmax pra transformar os a entrada em probabilidades
        self.a2 = softmax(self.z2)
        return self.a2


    def backward(self, X, y):
        # Propragação dos resultados para ajustar a rede!
        m = y.shape[0]  # Esse é o número de exemplos dentro do batch
        # A matriz y_one_hot transforma os rótulos em uma matriz de classes
        # onde cada linha representa um exemplo e cada coluna uma classe.
        y_one_hot = np.zeros((m, 10))
        # e depois essa linha coloca 1 na coluna correta da classe de cada
        # exemplo (3 = 0001000000)
        y_one_hot[np.arange(m), y] = 1
        # Esse é o erro da saída, previsão - verdade
        dz2 = self.a2 - y_one_hot
        # Esses são os gradientes para peso e bias da segunda camada
        dw2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        # Calculo do erro de cada neurônio da camada oculta
        dz1 = dz2 @ self.w2.T * relu_derivada(self.z1)
        # Variação dos pesos da camada oculta
        dw1 = X.T @ dz1 / m
        # Calculo do gradiente do bias da camada oculta
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Atualiza pesos, subtraindo o gradiente vezes a taxa de aprendizado,
        # melhorando as previsões
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1

        # processo conhecido como Gradient Descent, onde se ajusta os parâmetros
        # na direção oposta ao gradiente para minimizar a função de perda.
        # Isso é feito para cada camada da rede neural, ajustando os pesos e biases
        # de forma a melhorar as previsões da rede.

    def treinar(self, X, y, epocas, batch_size=64):
        # Ela recebe os dados, rótulos, num epocas e tam batch
        historico_perda = []  # Guarda a perda de cada época
        for epoca in range(epocas):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)  # embaralhando os dados
            X, y = X[indices], y[indices]  # Reorganizando dados e rótulos

            # o embaralhamento dos dados é crucial para evitar viés no treinamento
            # e garantir que a rede aprenda de forma mais robusta.
            # Isso garante que a rede neural não aprenda padrões específicos

            # Treinamento em conjunto de dados (batches)
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                predicoes = self.forward(X_batch)  # passo foward
                self.backward(X_batch, y_batch)  # passo backward

            # Calculo da perda total após cada época ser analisada
            # A função cross_entropy calcula a perda entre as previsões e os rótulos reais
            # A perda é uma medida de quão bem a rede está se saindo em relação aos rótulos reais.
            predicoes = self.forward(X)
            perda = cross_entropy(predicoes, y)
            historico_perda.append(perda)
            print(f"Época {epoca+1}/{epocas} - Perda: {perda:.8f}")
        return historico_perda
