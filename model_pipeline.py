import argparse
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from elasticsearch import Elasticsearch
import datetime

# Connect to Elasticsearch
es = Elasticsearch(
    "http://localhost:9200", timeout=30, max_retries=10, retry_on_timeout=True
)


def log_to_elasticsearch(run_id, metrics):
    """
    Logs the model's metrics to Elasticsearch.
    """
    doc = {
        "run_id": run_id,
        **metrics,
        "timestamp": datetime.datetime.now(),
    }
    es.index(index="mlflow-metrics", body=doc)


def load_data(file_path):
    """
    Load data from a CSV file.
    """
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df, target_column):
    """
    Preprocess the data by dropping unnecessary columns, encoding categorical features, and scaling numerical features.
    """
    df = df.drop(
        columns=[
            "State",
            "Area code",
            "Total day charge",
            "Total eve charge",
            "Total night charge",
            "Total intl charge",
        ],
        errors="ignore",
    )
    df = df.dropna()

    if df[target_column].dtype == "object":
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])

    categorical_features = ["International plan", "Voice mail plan"]
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X[X.select_dtypes(include=["float64", "int64"]).columns] = scaler.fit_transform(
        X.select_dtypes(include=["float64", "int64"])
    )

    return X, y, scaler


def train_model(X, y, kernel='rbf', random_state=42):
    """
    Train a Support Vector Machine (SVM) model and evaluate its performance.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(report)

    return model, accuracy, precision, recall, f1, report, X_test


def save_model(
    model, scaler, model_path="trained_model.joblib", scaler_path="scaler.joblib"
):
    """
    Save the trained model and scaler to disk.
    """
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


def preprocess_test_data(df, scaler, target_column):
    """
    Preprocess the test data using the same scaler used for training.
    """
    df = df.drop(
        columns=[
            "State",
            "Area code",
            "Total day charge",
            "Total eve charge",
            "Total night charge",
            "Total intl charge",
        ],
        errors="ignore",
    )
    df = df.dropna()

    if df[target_column].dtype == "object":
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])

    categorical_features = ["International plan", "Voice mail plan"]
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X[X.select_dtypes(include=["float64", "int64"]).columns] = scaler.transform(
        X.select_dtypes(include=["float64", "int64"])
    )

    return X, y


def evaluate_model_on_test_data(model, scaler, test_file_path, target_column):
    """
    Evaluate the model on the test data.
    """
    df = load_data(test_file_path)
    X_test, y_test = preprocess_test_data(df, scaler, target_column)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(report)

    return accuracy, precision, recall, f1, report


def main(args):
    """
    Main function to preprocess data, train the model, and log metrics to MLflow and Elasticsearch.
    """
    # Set the MLflow tracking URI and artifact location
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Churn Prediction")

    file_path = args.file_path
    test_file_path = "testData.csv"
    target_column = args.target_column
    random_state = args.random_state
    kernel = args.kernel

    if args.task in ["preprocess", "all"]:
        print("Loading and preprocessing data...")
        df = load_data(file_path)
        X, y, scaler = preprocess_data(df, target_column)
        print("Data preprocessing completed.")
        joblib.dump((X, y, scaler), "preprocessed_data.joblib")
        print("Preprocessed data saved to preprocessed_data.joblib")

    if args.task in ["train", "all"]:
        try:
            X, y, scaler = joblib.load("preprocessed_data.joblib")
            print("Preprocessed data loaded.")
        except FileNotFoundError:
            print("Error: Preprocessed data not found. Run preprocessing first.")
            return
        with mlflow.start_run() as run:
            model, accuracy, precision, recall, f1, report, X_test = train_model(
                X, y, kernel=kernel, random_state=random_state
            )

            # Log parameters and metrics to MLflow
            mlflow.log_param("kernel", kernel)
            mlflow.log_param("random_state", random_state)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Infer signature and log model properly
            signature = infer_signature(X_test, model.predict(X_test))
            mlflow.sklearn.log_model(
                model, "model", signature=signature, input_example=X_test.iloc[:5]
            )

            save_model(model, scaler)

            # Log to Elasticsearch
            log_to_elasticsearch(run.info.run_id, {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "report": report,
            })
        print("Model training completed.")

    if args.task in ["test", "all"]:
        try:
            model = joblib.load("trained_model.joblib")
            scaler = joblib.load("scaler.joblib")
            print("Trained model and scaler loaded.")
        except FileNotFoundError:
            print("Error: Trained model or scaler not found. Run training first.")
            return
        with mlflow.start_run() as run:
            accuracy, precision, recall, f1, report = evaluate_model_on_test_data(
                model, scaler, test_file_path, target_column
            )

            # Log metrics to MLflow
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1_score", f1)

            # Log to Elasticsearch
            log_to_elasticsearch(run.info.run_id, {
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1_score": f1,
                "test_report": report,
            })
        print("Model testing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Pipeline")
    parser.add_argument(
        "task",
        choices=["preprocess", "train", "test", "all"],
        help="Specify the task: 'preprocess', 'train', 'test', or 'all'",
    )
    parser.add_argument("--file_path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--target_column", type=str, required=True, help="Name of the target column")
    parser.add_argument("--test_size", type=float, default=0.2, help="Size of the test set")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--kernel", type=str, default="rbf", help="Kernel type for SVM")
    args = parser.parse_args()
    main(args)