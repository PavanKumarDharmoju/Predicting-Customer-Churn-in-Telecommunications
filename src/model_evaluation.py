from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import shap
import pandas as pd
import matplotlib.pyplot as plt
import pickle

class ModelEvaluator:
    def __init__(self, model):
        """
        Initialize the ModelEvaluator with a trained model.

        Parameters:
        model: The trained model object (e.g., RandomForestClassifier).
        """
        self.model = model

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Evaluate the model using standard metrics like classification report, AUC-ROC, and confusion matrix.

        Parameters:
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): True labels for the test set.

        Returns:
        None
        """
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # Classification report
        print("Classification Report:\n", classification_report(y_test, y_pred))

        # AUC-ROC score
        auc_roc = roc_auc_score(y_test, y_proba)
        print(f"AUC-ROC: {auc_roc}")

        # Confusion Matrix
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    def explain_with_shap(self, X_test: pd.DataFrame):
        """
        Explain model predictions using SHAP values.

        Parameters:
        X_test (pd.DataFrame): Test feature matrix.

        Returns:
        None
        """
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values[1], X_test, plot_type='bar')

# Example usage:
if __name__ == "__main__":
    # Load the trained model and preprocessed test dataset
    with open('../models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    df_test = pd.read_csv('../data/telecom_churn_preprocessed.csv')
    
    X_test = df_test.drop('Churn', axis=1)
    y_test = df_test['Churn']
    
    evaluator = ModelEvaluator(model)
    evaluator.evaluate(X_test, y_test)
    evaluator.explain_with_shap(X_test)
