# mlops_pipeline/scripts/02_data_preprocessing.py
import os, mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(test_size=0.25, random_state=42):
    mlflow.set_experiment("Titanic - Data Preprocessing")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("ml.step", "data_preprocessing")

        df = pd.read_csv("titanic/train.csv")

        # เลือกฟีเจอร์หลัก ๆ และทำความสะอาด
        use_cols = ["Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
        df = df[use_cols].copy()

        # เติมค่า missing
        df["Age"] = df["Age"].fillna(df["Age"].median())
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

        # เข้ารหัสตัวแปรหมวดหมู่
        df["Sex"] = (df["Sex"] == "male").astype(int)  # male=1, female=0
        df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

        # เปลี่ยนชื่อ target
        df = df.rename(columns={"Survived": "target"})

        X = df.drop("target", axis=1)
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        os.makedirs("processed_data", exist_ok=True)
        pd.concat([X_train, y_train], axis=1).to_csv("processed_data/train.csv", index=False)
        pd.concat([X_test, y_test], axis=1).to_csv("processed_data/test.csv", index=False)

        mlflow.log_param("test_size", test_size)
        mlflow.log_metric("training_set_rows", len(X_train))
        mlflow.log_metric("test_set_rows", len(X_test))
        mlflow.log_artifacts("processed_data", artifact_path="processed_data")

        # ส่ง run_id ให้ GitHub Actions
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"run_id={run_id}", file=f)

        print(f"[OK] Preprocessing Run ID: {run_id}")

if __name__ == "__main__":
    preprocess_data()
