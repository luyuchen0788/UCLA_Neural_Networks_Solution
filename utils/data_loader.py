import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    logging.info("Loading data from Excel file.")
    df = pd.read_excel(filepath)
    return df

def preprocess_data(df):
    logging.info("Starting data preprocessing.")
    df = df.drop(columns=["Serial_No"])
    df["Admit"] = (df["Admit_Chance"] > 0.75).astype(int)
    df = df.drop(columns=["Admit_Chance"])

    X = df.drop("Admit", axis=1)
    y = df["Admit"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    logging.info("Data standardized and scaler saved.")
    return X_scaled, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)