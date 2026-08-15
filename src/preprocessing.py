import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    
    # Fix TotalCharges — has spaces in original dataset
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Survival analysis needs two special columns:
    # duration  → how long the customer was observed (tenure in months)
    # event     → did the event (churn) actually happen? 1=yes, 0=censored
    df['duration'] = df['tenure']
    df['event']    = (df['Churn'] == 'Yes').astype(int)
    
    # Encode categoricals
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ['customerID', 'Churn']]
    
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Drop original churn and customerID
    df.drop(columns=['customerID', 'Churn', 'tenure'], inplace=True)
    
    return df

def get_survival_format(df: pd.DataFrame):
    """Returns (X, durations, events) for lifelines models"""
    X = df.drop(columns=['duration', 'event'])
    durations = df['duration']
    events    = df['event']
    return X, durations, events