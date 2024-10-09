# model_training.py

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pickle

# Import custom modules
from data_preprocessing import preprocess_data
from feature_engineering import FeatureEngineer

class ModelTrainer:
    def __init__(self):
        """
        Initialize the ModelTrainer with a RandomForestClassifier and hyperparameter grid.
        """
        self.model = RandomForestClassifier(random_state=42)
        self.param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        self.grid_search = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the Random Forest model using GridSearchCV for hyperparameter tuning.

        Parameters:
        X (pd.DataFrame): Input feature matrix.
        y (pd.Series): Target variable.

        Returns:
        Best model from GridSearchCV.
        """
        self.grid_search = GridSearchCV(self.model, self.param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        self.grid_search.fit(X, y)
        return self.grid_search.best_estimator_

    def save_model(self, model, file_path='../models/best_model.pkl'):
        """
        Save the trained model to a file.

        Parameters:
        model (sklearn model): Trained model object.
        file_path (str): Path to save the model. Default is '../models/best_model.pkl'.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {file_path}")

# Main workflow
if __name__ == "__main__":
    # Load and preprocess the dataset
    df = pd.read_csv('../data/telecom_churn.csv')
    
    # Step 1: Data Preprocessing
    df_preprocessed = preprocess_data(df)

    # Step 2: Feature Engineering
    # Initialize FeatureEngineer and create interaction features
    engineer = FeatureEngineer()
    df_engineered = engineer.create_interaction_features(df_preprocessed)

    # Define features and target variable
    X = df_engineered.drop('Churn', axis=1)
    y = df_engineered['Churn']

    # Step 3: Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Model Training
    trainer = ModelTrainer()
    best_model = trainer.train(X_train, y_train)

    # Step 5: Model Evaluation
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Print the classification report and AUC-ROC score
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba)}")

    # Step 6: Save the trained model
    trainer.save_model(best_model)
