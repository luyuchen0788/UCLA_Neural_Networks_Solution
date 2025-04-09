import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    df = pd.read_excel(filepath)
    return df

def split_data(df):
    X = df.drop("Admit", axis=1)
    y = df["Admit"]
    return train_test_split(X, y, test_size=0.2, random_state=42)