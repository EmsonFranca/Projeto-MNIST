import numpy as np

def relu(x):
    # Retorna o elemento se for positivo, caso contrário retorna zero
    return np.maximum(0, x)


def relu_derivada(x):
    # Retorna 1 para elementos positivos e 0 para negativos (como float)
    return (x > 0).astype(float)


def softmax(x):
    # Calcula a exponencial dos valores de entrada normalizados pelo máximo
    # para evitar overflow, depois normaliza para obter distribuição de probabilidade
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)