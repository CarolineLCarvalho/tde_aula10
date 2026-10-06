import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# k-means


iris = load_iris()

X = iris.data
y = iris.target

print("Formato dos dados:", X.shape)
print("Nomes das características:", iris.feature_names)

# K = 2
kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters_2 = kmeans_2.fit_predict(X)

# K = 3
kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters_3 = kmeans_3.fit_predict(X)

print("\nQuantidade de clusters:")
print("K = 2:", len(set(clusters_2)))
print("K = 3:", len(set(clusters_3)))

# Gráfico K = 2
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 2], X[:, 3], c=clusters_2, cmap="viridis")
plt.xlabel("Comprimento da pétala")
plt.ylabel("Largura da pétala")
plt.title("Clusters encontrados pelo K-Means - K=2")
plt.show()

# Gráfico K = 3
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 2], X[:, 3], c=clusters_3, cmap="viridis")
plt.xlabel("Comprimento da pétala")
plt.ylabel("Largura da pétala")
plt.title("Clusters encontrados pelo K-Means - K=3")
plt.show()

# Classes reais
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 2], X[:, 3], c=y, cmap="viridis")
plt.xlabel("Comprimento da pétala")
plt.ylabel("Largura da pétala")
plt.title("Classes reais do dataset Iris")
plt.show()

# Método do cotovelo
inertias = []

for k in range(1, 7):
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    modelo.fit(X)
    inertias.append(modelo.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 7), inertias, marker="o")
plt.xlabel("Número de clusters (K)")
plt.ylabel("Inertia")
plt.title("Método do Cotovelo")
plt.show()


# =========================
# PARTE 2 - KNN
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

modelo = KNeighborsClassifier(n_neighbors=5)

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

# Matriz de confusão
matriz = confusion_matrix(y_test, y_pred)

print("\nMatriz de confusão:")
print(matriz)

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=iris.target_names
)

disp.plot(cmap="Blues")
plt.title("Matriz de Confusão")
plt.show()

# Métricas
acuracia = accuracy_score(y_test, y_pred)

precisao = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\nMétricas:")
print("Acurácia:", acuracia)
print("Precisão:", precisao)
print("Recall:", recall)
print("F1-score:", f1)

print("\nRelatório de classificação:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))