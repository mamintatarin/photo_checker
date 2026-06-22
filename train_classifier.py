import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from face_utils import extract_face_embeddings, get_face_analysis_instance


def load_dataset(dataset_path):
    """
    Загрузка датасета из папок positive и negative
    """
    dataset_path = Path(dataset_path)
    positive_path = dataset_path / "positive"
    negative_path = dataset_path / "negative"

    if not positive_path.exists() or not negative_path.exists():
        raise ValueError(
            "Dataset folders 'positive' and 'negative' must exist in the dataset path"
        )

    face_analysis = get_face_analysis_instance()

    X = []  # Embeddings
    y = []  # Labels (1 for positive, 0 for negative)

    # Load positive samples
    for img_path in positive_path.glob("*.[jp][np]g"):  # *.jpg or *.png
        embeddings = extract_face_embeddings(img_path, face_analysis)
        if embeddings:
            for embedding in embeddings:
                X.append(embedding)
                y.append(1)  # Positive class

    # Load negative samples
    for img_path in negative_path.glob("*.[jp][np]g"):  # *.jpg or *.png
        embeddings = extract_face_embeddings(img_path, face_analysis)
        if embeddings:
            for embedding in embeddings:
                X.append(embedding)
                y.append(0)  # Negative class

    return np.array(X), np.array(y)


def train_classifier(x: np.typing.NDArray, y: np.typing.NDArray) -> MLPClassifier:
    """
    Обучение бинарного классификатора
    """
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    classifier_eval = MLPClassifier()
    classifier_eval.fit(train_x, train_y)
    preds = classifier_eval.predict(test_x)
    accuracy_test = accuracy_score(test_y, preds)
    print('Test accuracy:', accuracy_test)
    report = classification_report(test_y, preds)
    print(report)


    classifier = MLPClassifier()
    classifier.fit(x, y)

    # Evaluate on training data
    predictions = classifier.predict(x)
    accuracy = accuracy_score(y, predictions)
    print(f"Training accuracy: {accuracy:.4f}")

    return classifier


def main():
    parser = argparse.ArgumentParser(
        description="Train a binary classifier for face matching"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset folder containing positive and negative subfolders",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="face_classifier.pkl",
        help="Path to save the trained model",
    )

    args = parser.parse_args()

    print(f"Loading dataset from: {args.dataset_path}")
    X, y = load_dataset(args.dataset_path)

    print(f"Loaded {len(X)} face embeddings")
    if len(np.unique(y)) < 2:
        raise ValueError("Dataset must contain samples from both classes")

    print("Training classifier...")
    classifier = train_classifier(X, y)

    print(f"Saving classifier to: {args.output_model}")
    with open(args.output_model, "wb") as f:
        pickle.dump(classifier, f)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
