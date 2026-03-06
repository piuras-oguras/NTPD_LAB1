import joblib
import pandas as pd

# === Zadanie 3 cd. ===
model = joblib.load('model.joblib') # wczytywanie modelu z pliku

print("Model wczytany")

# przykładowe dane wejsciowe do predykcji (pierwszy wiersz z iris.data)
example = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],columns=["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"])

prediction = model.predict(example) # wykonanie predykcji na podstawie przykładowych danych-powinna być klasa 0
print(f"Klasa: {prediction}")