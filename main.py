import pandas as pd
import joblib

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix , classification_report, f1_score

# == Zadanie 1 ==

iris = load_iris() # pobranie zbioru danych Iris z scikit-learn

df = pd.DataFrame(iris.data, columns=iris.feature_names) # konwersja zbioru danych na DataFrame kolumny zawierają nazwy cech

df["target"] = iris.target # dodanie kolumny z etykietami irisów

# podstawowe statystyki danych
print(f"Kilka pierwszych wierszy: \n {df.head()} \n")
print(f"Informacja na temat rozmiaru danych oraz typach danych w kolumnach : \n{df.info()}  \n")
print(f"Statystyka kolumn: \n {df.describe()}  \n")

# === Zadanie 2 ===
X = df.drop("target", axis=1) # tworzę macierz cech poprzez usunięcie kolumny z etykietami ('target')
y = df["target"] # kolumna z etykietami ('target')

# podział zbioru na treningowy i testowy w proporcjach 80%/20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000) # stworzenie modelu regresji logistycznej
model.fit(X_train, y_train) # trening modelu
y_pred = model.predict(X_test) # przewidywanie etykiet

accuracy = accuracy_score(y_test, y_pred) # dokładność modelu
precision = precision_score(y_test, y_pred, average="macro") # precyzja
recall = recall_score(y_test, y_pred, average="macro") # czułość
f1 = f1_score(y_test, y_pred, average="macro") # F1-score

print(f"Dokładność: {accuracy}")
print(f"Precyzja: {precision}")
print(f"Czułość: {recall}")
print(f"F1-score: {f1}")
print(f"Raport klasyfikacji: \n {classification_report(y_test, y_pred)}")
print(f"Macierz pomyłek : \n {confusion_matrix(y_test, y_pred)}")

# === Zadanie 3 ===
joblib.dump(model, "model.joblib")  # zapisanie wytrenowanego modelu do pliku
