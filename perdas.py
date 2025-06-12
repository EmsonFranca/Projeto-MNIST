import numpy as np

def cross_entropy(predicoes, rotulos):
    # A função de perda avalia o quão distante a previsão da rede está do valor correto.
    # A variável m representa a quantidade de exemplos. A matriz log_probs armazena os
    # logaritmos das probabilidades previstas para as classes corretas. A adição de 1e-9
    # evita erro ao calcular o log de zero. O valor retornado é a média da perda total:
    # quanto menor esse valor, melhor o desempenho da rede.

    m = rotulos.shape[0]
    log_probs = -np.log(predicoes[range(m), rotulos] + 1e-9)
    return np.sum(log_probs) / m