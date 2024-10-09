# feature_engineering.py

import pandas as pd

class FeatureEngineer:
    def __init__(self):
        """
        Initialize the FeatureEngineer for creating interaction terms or other feature engineering tasks.
        """
        pass

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between selected numerical columns.

        Parameters:
        df (pd.DataFrame): The input dataframe.

        Returns:
        pd.DataFrame: Dataframe with new interaction features.
        """
        # Create interaction terms as new features
        df['Monthly_Tenure_Interaction'] = df['MonthlyCharges'] * df['tenure']
        df['Total_Tenure_Interaction'] = df['TotalCharges'] * df['tenure']

        return df
