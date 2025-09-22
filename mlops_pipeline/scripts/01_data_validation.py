# mlops_pipeline/scripts/01_data_validation.py
import mlflow, pandas as pd

def validate_data():
    mlflow.set_experiment("Titanic - Data Validation")
    with mlflow.start_run():
        mlflow.set_tag("ml.step", "data_validation")
        df = pd.read_csv("titanic/train.csv")
        assert "Survived" in df.columns, "Missing target column 'Survived'"

        num_rows, num_cols = df.shape
        num_classes = df["Survived"].nunique()
        missing_values = df.isnull().sum().sum()

        print(f"Dataset shape: {num_rows} rows, {num_cols} columns")
        print(f"Classes: {num_classes} | Missing values: {missing_values}")

        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("num_cols", num_cols)
        mlflow.log_metric("missing_values", int(missing_values))
        mlflow.log_param("num_classes", int(num_classes))

        status = "Success" if missing_values == 0 and num_classes >= 2 else "Failed"
        mlflow.log_param("validation_status", status)
        print(f"Validation status: {status}")

if __name__ == "__main__":
    validate_data()
