# Execução do modelo de classificação de doenças cardíacas
Criação e execução de modelos KNN e SVM para a classificação de doenças cardíacas em homens da África do Sul.

Para criar os modelos de classificação, [este](https://blog.goodaudience.com/heart-disease-prediction-aa656f2db585) artigo, criado com base [neste repositório](https://github.com/sahilverma0696/heart-disease-prediction) foi utilizado como referência.

## Treinamento
O treinamento dos modelos se dá com a execução das celulas do script de main.ipynb contido na pasta model_train. Para executar o script é necessário abrir um terminal e executar o comando:

```sh
$ jupyter notebook main.ipynb
```

Os modelos KNN e SVM serão criados na mesma pasta que contém main.ipynb. 

## Execução
O script model_run.py é o responsável por executar os modelos utilizando novos dados de entrada. Em um terminal, execute o comando "python model_run.py" passando como parâmetros os dados de entrada, o caminho do modelo (KNN, SVM, ou algum outro que tenha sido criado) e o nome do arquivo de saída.

Os parâmetros são descritos como:
| Parâmetro | Descrição |
| ------ | ------ |
| -i / --input | caminho para os dados de entrada |
| -m / --model | caminho para o modelo a ser executado |
| -o / --output | caminho para o resultado da classificação |

Exemplo:
```sh
$ python model_run.py -i test_data.csv -m model_train/knn.joblib -o results.csv
```
