import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_and_preprocess_data(data_path: str, seq_len: int = 5):
    """
    Loads data from a Parquet file and preprocesses it to return sequences and targets
    for RNN training.

    Parameters:
        data_path (str): Path to the Parquet file.
        seq_len (int): Length of the input sequences.

    Returns:
        X (np.ndarray): Input sequences for the RNN, shape (samples, seq_len, features).
        y (np.ndarray): Target values, shape (samples,).
    """

    # Load data
    df = pd.read_parquet(data_path)

    # Filter patients with >= seq_len + 1 measurements
    valid_patients = df["PatientID"].value_counts()
    valid_patients = valid_patients[valid_patients >= seq_len + 1].index
    df = df[df["PatientID"].isin(valid_patients)].copy()

    # Sort chronologically
    df = df.sort_values(by=["PatientID", "Timestamp"])

    # Drop irrelevant or leakage-prone columns
    drop_cols = ["Unnamed: 0", "PatientId", "PatientID", "ResultUnit", "IsResultValueTooHigh",
                 "IsResultValueTooLow", "GlucoseLevel", "ShipToAccountId"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Boolean to int
    bool_cols = ["IsControlSolutionDetected", "IsEmergencyTest"]
    for col in bool_cols:
        df[col] = df[col].astype(int)

    # One-Hot-Encoding
    df = pd.get_dummies(df, columns=["MeasurementStatus", "Location", "Region"], drop_first=True)

    # Frequency Encoding for high-cardinality IDs
    for col in ["DeviceId", "StripId", "InstrumentLotId", "Institution"]:
        freq = df[col].value_counts(normalize=True)
        df[col + "_freq"] = df[col].map(freq)
        df.drop(columns=col, inplace=True)

    # Timestamp features
    df["hour"] = df["Timestamp"].dt.hour
    df["dayofweek"] = df["Timestamp"].dt.dayofweek
    df["month"] = df["Timestamp"].dt.month

    # Cyclical encoding for hour and dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df = df.drop(columns=["Timestamp", "hour", "dayofweek", "month"])

    # TimeSinceLast NaN → large value (assume no prior test)
    df["TimeSinceLast"] = df["TimeSinceLast"].fillna(1e6)

    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=["number"]).columns.difference(["GlucoseValue"])
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Build sequences per patient
    X, y = [], []
    for _, group in df.groupby("DeviceId_freq"):
        values = group.to_numpy()
        if len(values) >= seq_len + 1:
            for i in range(len(values) - seq_len):
                X.append(values[i:i+seq_len, :-1])  # all features except GlucoseValue
                y.append(values[i+seq_len, -1])     # GlucoseValue as target

    return np.array(X), np.array(y)


if __name__ == "__main__":
    data_path = os.path.join("data", "phoenix_anonymized.parquet")
    X, y = load_and_preprocess_data(data_path)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
