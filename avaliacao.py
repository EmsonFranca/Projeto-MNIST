# Módulo responsável por avaliar o desempenho da rede neural durante e após o treinamento.

import numpy as np

def calcular_acuracia(predicoes, rotulos):
    # Calcula a acurácia da rede neural comparando as previsões com os rótulos reais.
    # np.argmax retorna o índice da maior probabilidade (classe predita) para cada exemplo.
    # O parâmetro axis=1 indica que a busca será feita em cada linha (por exemplo).
    # A igualdade entre rótulos previstos e reais resulta em um array de valores booleanos.
    # np.mean converte esse array em uma proporção de acertos sobre o total de exemplos.
    pred_labels = np.argmax(predicoes, axis=1)
    return np.mean(pred_labels == rotulos)

def matriz_confusao(rotulos_verdadeiros, rotulos_preditos, num_classes=10):
    # Gera uma matriz de confusão que mostra os acertos e erros da rede por classe.
    # Cada linha representa a classe real, e cada coluna representa a classe predita.
    # Inicialmente a matriz é preenchida com zeros.
    # Para cada par (verdadeiro, predito), incrementa a célula correspondente.
    # Exemplo: Se verdadeiro = 3 e predito = 1 → matriz[3][1] += 1 (erro).
    matriz = np.zeros((num_classes, num_classes), dtype=int)
    for verdadeiro, predito in zip(rotulos_verdadeiros, rotulos_preditos):
        matriz[verdadeiro, predito] += 1
    return matriz

def exibir_matriz_confusao(matriz):
    # Exibe no terminal a matriz de confusão gerada.
    # A primeira linha imprime os rótulos preditos.
    # As linhas seguintes mostram os valores reais e as contagens correspondentes.
    print("Matriz de Confusão:")
    print("    " + " ".join(f"{i:^5}" for i in range(matriz.shape[1])))
    for i, linha in enumerate(matriz):
        print(f"{i}: " + " ".join(f"{val:^5}" for val in linha))
