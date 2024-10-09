# Telecom Customer Churn Prediction

## Project Overview

This project aims to predict customer churn for a telecommunications company using advanced machine learning techniques. By identifying customers who are likely to leave, the company can take preventive measures to retain them, reducing churn and increasing profitability.

## Problem Statement

Customer churn is a major challenge for telecom companies, and predicting which customers are likely to churn is critical for business retention strategies. This project focuses on building a machine learning model that can predict whether a customer will churn based on customer demographics, services used, and payment details.

---

## Dataset

- **Source**: [Kaggle Telecom Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Description**: The dataset contains information about customer demographics, account details, and the services they are subscribed to. The target variable `Churn` indicates whether a customer has left the company or not.

### Features:
1. **CustomerID**: Unique ID for each customer (this feature will be dropped as it is not useful for prediction).
2. **Demographic Features**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
3. **Account Features**: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
4. **Services Subscribed**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

---

## Project Workflow

1. **Data Preprocessing**
    - **Handling Missing Values**: The `TotalCharges` column contains some non-numeric values. These values are coerced into `NaN` and then filled with the median.
    - **Dropping Unnecessary Columns**: The `customerID` column is dropped as it doesn't provide any useful information for model training.
    - **Encoding Categorical Features**: Binary categorical columns like `gender`, `Partner`, etc., are label-encoded. Multi-category columns like `InternetService`, `Contract`, etc., are one-hot encoded.
    - **Scaling**: Numerical features (`MonthlyCharges` and `TotalCharges`) are standardized using `StandardScaler`.

2. **Feature Engineering**
    - **Interaction Terms**: Interaction terms such as `MonthlyCharges * tenure` and `TotalCharges * tenure` are created to capture additional relationships between features.

3. **Model Training**
    - The dataset is split into training and test sets.
    - A **Random Forest Classifier** is used as the primary model, with hyperparameter tuning using **GridSearchCV**.
    - **Hyperparameters** such as the number of estimators (`n_estimators`), maximum tree depth (`max_depth`), and the minimum number of samples for splitting (`min_samples_split`) are optimized.
  
4. **Model Evaluation**
    - The model is evaluated using several metrics, including:
        - **Classification Report**: Provides precision, recall, F1-score, and accuracy.
        - **AUC-ROC Score**: Measures the model's ability to distinguish between churners and non-churners.

5. **Model Saving**
    - The best model is saved to disk in the `models/` directory for future use.

---

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Required libraries: `pandas`, `scikit-learn`, `pickle`

You can install the required libraries using:
```bash
pip install -r requirements.txt
```

**Directory Structure**
```bash
Machine_Learning_Project/
│
├── data/                   # Raw and processed data
│   └── telecom_churn.csv    # Dataset
├── models/                 # Directory to save trained models
├── src/                    # Source code for data preprocessing, feature engineering, model training
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── model_training.py
├── README.md               # Project documentation
├── requirements.txt        # Dependencies
```

**Running the Project**

* Clone the Repository

    Clone the repository to your local machine:
    ```bash
    git clone https://github.com/your-username/telecom-churn-prediction.git
    cd telecom-churn-prediction
    ```
* Prepare the Dataset

    Ensure that the dataset (telecom_churn.csv) is located in the data/ directory. If you do not have the dataset, download it from Kaggle.

* Run Model Training

    To train the model and generate predictions, run the model_training.py script:

    ```bash
    python src/model_training.py
    ```

This script will:

- Load and preprocess the dataset.
Perform feature engineering (create interaction features).
- Train a Random Forest model with hyperparameter tuning.
Evaluate the model and print the classification report and AUC-ROC score.
- Save the best model in the models/ directory.

**Model Evaluation and Results**
- Once the model is trained, the following evaluation metrics will be printed:

- Classification Report:

Precision, recall, F1-score, and accuracy for both classes (churn and non-churn).

Example output:

```text

Classification Report:
               precision    recall  f1-score   support

        0       0.85      0.92      0.88      1200
        1       0.72      0.55      0.63       400

accuracy                           0.83      1600
macro avg       0.78      0.74      0.75      1600
weighted avg    0.82      0.83      0.82      1600
```
**AUC-ROC Score:**

The AUC-ROC score measures how well the model separates churners from non-churners.

Example output:

```text
AUC-ROC: 0.85
```
**Model Saved:**

The best model will be saved in the models/ directory as best_model.pkl.
Example output:

```text
Model saved to ../models/best_model.pkl
```

**Next Steps and Future Improvements**

- Model Optimization: Additional hyperparameter tuning could further optimize model performance.
- Advanced Feature Engineering: Investigating domain-specific interaction terms or additional features could improve model accuracy.
- Model Deployment: Deploy the saved model using a Flask API for real-time prediction.
- Cross-Validation: Implement cross-validation to ensure robust model performance across different splits of the dataset.

**Contributing**

Feel free to contribute to the project! Please create a new branch for your changes and submit a pull request.


**Contact**

For any questions or suggestions, please contact dharmojupavankumar@gmail.com



