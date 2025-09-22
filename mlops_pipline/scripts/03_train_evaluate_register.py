import sys
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.artifacts import download_artifacts


def train_evaluate_register(preprocessing_run_id: str, C: float = 1.0) -> None:
    """
    Loads preprocessed data, trains a model, evaluates it,
    and registers the model in the MLflow Model Registry if it
    meets the performance threshold.
    """
    ACCURACY_THRESHOLD = 0.1
    mlflow.set_experiment("Wine Quality - Model Training")

    with mlflow.start_run(run_name=f"logistic_regression_C_{C}"):
        print(f"Starting training run with C={C}...")
        mlflow.set_tag("ml.step", "model_training_evaluation")
        mlflow.log_param("preprocessing_run_id", preprocessing_run_id)

        # 1) โหลดข้อมูลจาก Artifacts ของ Preprocessing Run
        try:
            # 1.1 ดาวน์โหลด artifacts มาที่ local ชั่วคราว
            local_artifact_path = download_artifacts(
                run_id=preprocessing_run_id,
                artifact_path="processed_data"
            )
            print(f"Artifacts downloaded to: {local_artifact_path}")

            # 1.2 พาธไฟล์ CSV
            train_path = os.path.join(local_artifact_path, "train.csv")
            test_path = os.path.join(local_artifact_path, "test.csv")

            # 1.3 อ่านไฟล์
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            print("Successfully loaded data from downloaded artifacts.")
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            print("Please ensure the preprocessing_run_id is correct.")
            sys.exit(1)

        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']

        # 2) Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(C=C, random_state=42, max_iter=10000))
        ])
        pipeline.fit(X_train, y_train)

        # 3) Evaluate
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")

        # 4) Log params/metrics/model
        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(pipeline, "wine_classifier_pipeline")

        # 5) Register model (ถ้าผ่าน threshold)
        if acc >= ACCURACY_THRESHOLD:
            print(f"Model accuracy ({acc:.4f}) meets the threshold. Registering model...")
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/wine_classifier_pipeline"
            registered_model = mlflow.register_model(model_uri, "wine-classifier-prod")
            print(f"Model registered as '{registered_model.name}' version {registered_model.version}")
        else:
            print(f"Model accuracy ({acc:.4f}) is below the threshold. Not registering.")

        print("Training run finished.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/03_train_evaluate_register.py <preprocessing_run_id> [C_value]")
        sys.exit(1)

    run_id = sys.argv[1]
    c_value = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    train_evaluate_register(preprocessing_run_id=run_id, C=c_value)
