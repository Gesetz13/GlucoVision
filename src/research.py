import os
import pandas as pd

# Basisverzeichnis = Ordner, in dem dieses Skript liegt (also src/)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Pfad zur Parquet-Datei im data-Unterordner
data_path = os.path.join(base_dir, "data", "phoenix_anonymized.parquet")

# Daten einlesen
df = pd.read_parquet(data_path)

# Testausgabe
print(df.head().to_dict)
