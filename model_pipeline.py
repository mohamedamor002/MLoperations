import argparse

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df, target_column):
    df = df.drop(columns=['State', 'Area code', 'Total day charge', 'Total eve charge',
                          'Total night charge', 'Total intl charge'], errors='ignore')
    df = df.dropna()

    if df[target_column].dtype == 'object':
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])

    categorical_features = ['International plan', 'Voice mail plan']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X[X.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(
        X.select_dtypes(include=['float64', 'int64']))

    return X, y, scaler


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy:.4f}')
    print(report)

    return model, accuracy, report, X_test


def save_model(model, scaler, model_path='trained_model.joblib',
               scaler_path='scaler.joblib'):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


def main(args):
    mlflow.set_tracking_uri("http://localhost:5004")  # Add this line
    file_path = "data.csv"
    target_column = "Churn"

    if args.task in ['preprocess', 'all']:
        print("Loading and preprocessing data...")
        df = load_data(file_path)
        X, y, scaler = preprocess_data(df, target_column)
        print("Data preprocessing completed.")
        joblib.dump((X, y, scaler), 'preprocessed_data.joblib')
        print("Preprocessed data saved to preprocessed_data.joblib")

    if args.task in ['train', 'all']:
        try:
            X, y, scaler = joblib.load('preprocessed_data.joblib')
            print("Preprocessed data loaded.")
        except FileNotFoundError:
            print("Error: Preprocessed data not found. Run preprocessing first.")
            return
        with mlflow.start_run():
            model, accuracy, report, X_test = train_model(X, y)

            mlflow.log_param("kernel", "rbf")
            mlflow.log_param("random_state", 42)
            mlflow.log_metric("accuracy", accuracy)

            # Infer signature and log model properly
            signature = infer_signature(X_test, model.predict(X_test))
            mlflow.sklearn.log_model(model,
                                     "model",
                                     signature=signature,
                                     input_example=X_test.iloc[:5])

            save_model(model, scaler)
        print("Model training completed.")

        print("Training the model...")
        model, accuracy, report, X_test = train_model(X, y)
        save_model(model, scaler)
        print("Model training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Pipeline")
    parser.add_argument(
        'task',
        choices=[
            'preprocess',
            'train',
            'all'],
        help="Specify the task: 'preprocess', 'train', or 'all'")
    args = parser.parse_args()
    main(args)
