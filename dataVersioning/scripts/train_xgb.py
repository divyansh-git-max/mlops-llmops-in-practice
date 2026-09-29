import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

def load_and_preprocess_data(data_path="data/churn.csv"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Please check the path.")

    df = pd.read_csv(data_path)
    
    # Drop customer ID as it is not a useful feature
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        
    # TotalCharges is sometimes read as an object due to blank spaces
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # Convert target variable to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # One-hot encode remaining categorical columns
    df = pd.get_dummies(df, drop_first=True)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    print("Training XGBoost model (Version 2)...")
    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=2.8, # Handles class imbalance typical in churn datasets
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"XGBoost - Test Accuracy: {acc:.4f}")
    print(f"XGBoost - Test F1 Score: {f1:.4f}")
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.pkl"
    joblib.dump(model, model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    main()
