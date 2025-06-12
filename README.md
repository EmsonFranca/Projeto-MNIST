# Classificação de Dígitos com MLP (MNIST)

Este projeto apresenta uma implementação completa de uma rede neural Perceptron Multicamadas (MLP) desenvolvida inteiramente em NumPy, sem o uso de frameworks de deep learning. O sistema foi projetado para classificação automática de dígitos manuscritos a partir do conhecido dataset MNIST, demonstrando os princípios fundamentais de redes neurais artificiais.

------------------------------------------------------------

## Tecnologias Utilizadas

- Python 3.12
- Pip 25.0
- NumPy
- Matplotlib
- Gzip / Struct

------------------------------------------------------------

## Como Executar

   1. Entra na pasta
      ~~~
      cd Projeto-MNIST
      ~~~
   3. Instale as dependências:
      ~~~
      pip install -r requirements.txt
      ~~~

   4. Verifique se os seguintes arquivos estão na pasta `dados/`:

      - `train-images-idx3-ubyte.gz`
      - `train-labels-idx1-ubyte.gz`
      - `t10k-images-idx3-ubyte.gz`
      - `t10k-labels-idx1-ubyte.gz`

      Você pode baixar os arquivos diretamente dos seguintes links:

      - [train-images-idx3-ubyte.gz](https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz)
      - [train-labels-idx1-ubyte.gz](https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz)
      - [t10k-images-idx3-ubyte.gz](https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz)
      - [t10k-labels-idx1-ubyte.gz](https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz)

      Esses arquivos são mantidos no repositório MNIST do _Department of Communications Engineering University of Paderborn_, disponível em: [fgnt/mnist](https://github.com/fgnt/mnist) no GitHub.

   5. Execute o script principal:
      ~~~
      python main.py
      ~~~
------------------------------------------------------------

## Estrutura do Projeto

```
Projeto-MNIST/
├── dados/
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
│
├── ativacoes.py
├── avaliacao.py
├── data_loader.py
├── mlp.py
├── perdas.py
├── visualizacao.py
├── main.py
│
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
```
------------------------------------------------------------

## Saídas do Projeto

- Acurácia final da rede no conjunto de teste.
- Matriz de confusão exibida no terminal.
- Gráfico da função de perda por época.
- Comparação com imagens entre a predição da rede e os arquivos originais.
