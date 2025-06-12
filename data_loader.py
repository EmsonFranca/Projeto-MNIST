# Módulo responsável por lidar com os dados do MNIST:
# inclui carregamento, descompactação, leitura e normalização das imagens e rótulos.

# Explicação dos imports:
# gzip   → permite abrir e ler arquivos comprimidos no formato .gz (usado pelo MNIST).
# os     → utilizado para manipular caminhos de arquivos de forma multiplataforma.
# struct → permite ler dados binários com estrutura específica (como o cabeçalho dos arquivos MNIST).
# numpy  → biblioteca fundamental para operações matemáticas e manipulação de arrays.

import gzip
import os
import struct
import numpy as np

# Função que carrega e normaliza as imagens do conjunto MNIST:
# - Abre o arquivo .gz em modo binário.
# - Lê o cabeçalho (16 bytes) contendo metadados: número de imagens, linhas e colunas.
# - Lê os dados restantes como uma sequência de bytes representando os pixels das imagens.
# - Reorganiza os dados em um array 2D (uma imagem por linha).
# - Normaliza os valores de 0 a 255 para um intervalo de 0 a 1 (float32),
#   o que ajuda na estabilidade e eficiência do treinamento da rede neural.
def carregar_imagens(caminho_arquivo):
    with gzip.open(caminho_arquivo, 'rb') as f:
        _, num_imagens, linhas, colunas = struct.unpack(">IIII", f.read(16))
        imagens = np.frombuffer(f.read(), dtype=np.uint8)
        imagens = imagens.reshape(num_imagens, linhas * colunas)
        imagens = imagens.astype(np.float32) / 255.0
        return imagens

# Função que carrega os rótulos das imagens:
# - Abre o arquivo .gz em modo binário.
# - Lê o cabeçalho (8 bytes) contendo metadados (ignorados aqui).
# - Lê o restante como rótulos (valores inteiros de 0 a 9).
# - Retorna um array contendo os rótulos correspondentes às imagens.
def carregar_rotulos(caminho_arquivo):
    with gzip.open(caminho_arquivo, 'rb') as f:
        _, _ = struct.unpack(">II", f.read(8))
        rotulos = np.frombuffer(f.read(), dtype=np.uint8)
        return rotulos

# Função principal que carrega os dados de treino e teste:
# - Concatena corretamente os caminhos para os arquivos de imagens e rótulos.
# - Usa as funções acima para carregar e preparar os arrays de entrada (X) e saída (y).
# - Retorna os dados prontos para o treinamento e teste da rede neural.
def carregar_dados_mnist(pasta="dados"):
    X_train = carregar_imagens(os.path.join(pasta, 'train-images-idx3-ubyte.gz'))
    y_train = carregar_rotulos(os.path.join(pasta, 'train-labels-idx1-ubyte.gz'))
    X_test = carregar_imagens(os.path.join(pasta, 't10k-images-idx3-ubyte.gz'))
    y_test = carregar_rotulos(os.path.join(pasta, 't10k-labels-idx1-ubyte.gz'))
    return X_train, y_train, X_test, y_test
