import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # For saving the scaler

def load_data(file_path):
    """Load the UCLA admission dataset (Excel format)"""
    df = pd.read_excel(file_path)
    return df

def preprocess_data(df, save_path='data/cleaned_admission.csv', scaler_path='data/scaler.pkl'):
    """Clean, transform, scale data, and save to a new file"""
    # Drop Serial_No column if present
    if 'Serial_No' in df.columns:
        df = df.drop(columns=['Serial_No'])

    # Handle missing values - fill with median for numerical columns, mode for categorical
    df.fillna(df.median(), inplace=True)  # Numeric columns
    df.fillna(df.mode().iloc[0], inplace=True)  # Categorical columns

    # Convert Admit_Chance into binary classification: 1 if > 0.8 else 0
    df['Admit_Chance'] = df['Admit_Chance'].apply(lambda x: 1 if x > 0.8 else 0)

    # Encode categorical variables
    df = pd.get_dummies(df, columns=['University_Rating', 'Research'], drop_first=True)

    # Split features and target
    X = df.drop(columns=['Admit_Chance'])
    y = df['Admit_Chance']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler for future use
    joblib.dump(scaler, scaler_path)

    # Save cleaned data
    cleaned_df = pd.DataFrame(X_scaled, columns=X.columns)
    cleaned_df['Admit_Chance'] = y.values
    cleaned_df.to_csv(save_path, index=False)

    return X_scaled, y

