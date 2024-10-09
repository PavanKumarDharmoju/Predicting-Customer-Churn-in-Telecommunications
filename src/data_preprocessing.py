# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the telecom churn dataset by handling missing values, converting data types,
    encoding categorical variables, and scaling numerical features.

    Parameters:
    df (pd.DataFrame): The raw dataframe.

    Returns:
    pd.DataFrame: Processed dataframe ready for model training.
    """

    # Drop the 'customerID' column as it's not useful for prediction
    df.drop('customerID', axis=1, inplace=True)

    # Convert 'TotalCharges' to numeric, forcing errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing 'TotalCharges' values (if any) with the median
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Convert 'Churn' to binary (Yes = 1, No = 0)
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # List of columns to encode with Label Encoding (binary columns)
    binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

    label_encoder = LabelEncoder()
    for column in binary_columns:
        df[column] = label_encoder.fit_transform(df[column])

    # One-Hot Encoding for categorical variables with more than two categories
    categorical_columns = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                           'Contract', 'PaymentMethod']

    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Scale the numerical columns
    scaler = StandardScaler()
    df[['MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['MonthlyCharges', 'TotalCharges']])

    return df
