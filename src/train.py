import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocessing import preprocess_training
from src.model import LogisticRegressionMBGD
from src.config import *
from src.utils import evaluate

def train():
    df = load_data()
    df, feature_columns, scaler = preprocess_training(df)

    X = df.drop("target", axis=1).values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = LogisticRegressionMBGD(
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    model.fit(X_train, y_train)

    # Evaluate on holdout
    y_pred = model.predict(X_test)
    acc, cm = evaluate(y_test, y_pred)

    # Persist model plus feature metadata for inference alignment
    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "scaler": scaler,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print(f"Model trained. Accuracy: {acc:.3f}")
    print("Confusion matrix:\n", cm)

if __name__ == "__main__":
    train()
