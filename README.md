 Atividade de Machine Learning
 K-Means e Avaliação de Classificadores

**Nome:** Caroline Lopes de Carvalho

## Parte 1 — K-Means

### Pergunta 1

O dataset Iris possui **150 amostras e 4 características**: comprimento da sépala, largura da sépala, comprimento da pétala e largura da pétala.

### Pergunta 2

Foram realizados dois experimentos:

* **K = 2:** foram encontrados 2 clusters.
* **K = 3:** foram encontrados 3 clusters.

### Pergunta 3

Os clusters encontrados pelo K-Means fazem sentido, principalmente quando usamos K = 3. O resultado fica mais parecido com as três classes reais do dataset Iris. A classe Setosa é mais fácil de separar, enquanto as outras duas ficam mais próximas. Por isso, K = 3 representa melhor a estrutura dos dados.

### Pergunta 4

O valor **K = 3** parece ser uma escolha razoável para esse dataset. No gráfico do método do cotovelo, depois desse valor a diminuição da inertia fica menor. Além disso, o dataset Iris possui três classes e o K = 3 conseguiu representar melhor os grupos.



## Parte 2 — Classificação com KNN

### Pergunta 5

Foi utilizada a proporção de **80% dos dados para treinamento e 20% para teste**.

O modelo escolhido foi o **K-Nearest Neighbors (KNN)** com `n_neighbors=5`.

### Pergunta 6

A matriz de confusão apresentou:

```text
[[10  0  0]
 [ 0 10  0]
 [ 0  0 10]]
```

Foram **30 previsões classificadas corretamente** e **nenhuma previsão incorreta**.

Nenhuma das três classes apresentou mais erros que as outras, pois todas tiveram 10 acertos.

### Métricas

| Métrica  | Resultado |
| -------- | --------: |
| Acurácia |       1.0 |
| Precisão |       1.0 |
| Recall   |       1.0 |
| F1-score |       1.0 |

---

## Cálculo manual — Precisão

Escolhi calcular a precisão da classe **Setosa**.

**Fórmula:**

```text
Precisão = VP / (VP + FP)
```

**Valores:**

```text
VP = 10
FP = 0
```

**Cálculo:**

```text
Precisão = 10 / (10 + 0)
Precisão = 10 / 10
Precisão = 1.0
```

O resultado foi **1.0 (100%)**, igual ao resultado apresentado pelo Scikit-learn.


## Pergunta 7 — Análise crítica

O modelo apresentou um desempenho equilibrado entre precisão e recall, pois os dois resultados foram iguais a 1.0. Para esse problema, as duas métricas são importantes porque queremos identificar corretamente as diferentes espécies de flores. O recall pode ser interessante para garantir que as flores de cada classe sejam identificadas corretamente. A acurácia sozinha não seria suficiente para avaliar um modelo em todos os casos, porque precisão, recall e F1-score também mostram informações sobre o desempenho. Neste caso, todas as métricas apresentaram resultados muito bons.



## Pergunta 8 — Supervisionado x Não Supervisionado

Na Parte 1, o K-Means foi utilizado sem os rótulos reais das flores, tentando encontrar os grupos sozinho. Por isso, ele é um algoritmo de aprendizado **não supervisionado**.

Na Parte 2, o KNN recebeu os rótulos das classes durante o treinamento e aprendeu a classificar novas amostras. Por isso, é um modelo de aprendizado **supervisionado**.



## Conclusão

Com essa atividade aprendi a utilizar o K-Means para encontrar grupos sem informar as classes para o algoritmo. Também consegui perceber como a escolha do número de clusters influencia o resultado. No modelo supervisionado, utilizei o KNN para classificar as flores usando dados de treinamento e teste. A matriz de confusão ajudou a visualizar os acertos e erros do modelo. Também entendi que não é bom analisar um classificador somente pela acurácia, pois outras métricas também mostram informações importantes. Nesse caso, o modelo apresentou resultados muito bons em todas as métricas.
