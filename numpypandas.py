import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CSV-Datei mit Pandas einlesen
df = pd.read_csv("hello/customers-500000.csv")

# Erste Zeilen des DataFrames anzeigen
print("Erste Zeilen des DataFrames:")
print(df.head())

# Numerische Spalten für die Demonstration auswählen
numeric_cols = df.select_dtypes(include='number').columns
print("\nAusgewählte numerische Spalten:")
print(numeric_cols)

# Numerische Spalten in ein NumPy-Array umwandeln
data = df[numeric_cols].to_numpy()

# Erste Zeilen des NumPy-Arrays anzeigen
print("\nErste Zeilen des NumPy-Arrays:")
print(data[:5])

# Einige grundlegende Operationen mit NumPy demonstrieren
print("\nGrundlegende Operationen mit NumPy:")

# Mittelwert jeder Spalte berechnen
mean_values = np.mean(data, axis=0)
print("Mittelwert jeder Spalte:")
print(mean_values)

# Standardabweichung jeder Spalte berechnen
std_values = np.std(data, axis=0)
print("Standardabweichung jeder Spalte:")
print(std_values)

# Summe jeder Spalte berechnen
sum_values = np.sum(data, axis=0)
print("Summe jeder Spalte:")
print(sum_values)

# Maximalwert jeder Spalte berechnen
max_values = np.max(data, axis=0)
print("Maximalwert jeder Spalte:")
print(max_values)

# Minimalwert jeder Spalte berechnen
min_values = np.min(data, axis=0)
print("Minimalwert jeder Spalte:")
print(min_values)

# Datenvisualisierung: Balkendiagramm für die Mittelwerte
plt.figure(figsize=(10, 6))
plt.bar(numeric_cols, mean_values, color='skyblue')
plt.title("Mittelwerte der numerischen Spalten", fontsize=16)
plt.xlabel("Spalten", fontsize=14)
plt.ylabel("Mittelwert", fontsize=14)
plt.xticks(rotation=45)
plt.show()

# Korrelationsmatrix berechnen
correlation_matrix = df[numeric_cols].corr()
print("\nKorrelationsmatrix:")
print(correlation_matrix)

# Heatmap der Korrelationsmatrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Korrelationsmatrix", fontsize=16)
plt.show()