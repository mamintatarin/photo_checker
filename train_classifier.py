import os
import argparse
import numpy as np
from pathlib import Path
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import pickle
import sys
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
        raise ValueError("Dataset folders 'positive' and 'negative' must exist in the dataset path")
    
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


def train_classifier(X, y):
    """
    Обучение бинарного классификатора
    """
    classifier = Perceptron(random_state=42, max_iter=1000, tol=1e-3)
    classifier.fit(X, y)
    
    # Evaluate on training data
    predictions = classifier.predict(X)
    accuracy = accuracy_score(y, predictions)
    print(f"Training accuracy: {accuracy:.4f}")
    
    return classifier


def main():
    parser = argparse.ArgumentParser(description='Train a binary classifier for face matching')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset folder containing positive and negative subfolders')
    parser.add_argument('--output-model', type=str, default='face_classifier.pkl', help='Path to save the trained model')
    
    args = parser.parse_args()
    
    print(f"Loading dataset from: {args.dataset_path}")
    X, y = load_dataset(args.dataset_path)
    
    print(f"Loaded {len(X)} face embeddings")
    if len(np.unique(y)) < 2:
        raise ValueError("Dataset must contain samples from both classes")
    
    print("Training classifier...")
    classifier = train_classifier(X, y)
    
    print(f"Saving classifier to: {args.output_model}")
    with open(args.output_model, 'wb') as f:
        pickle.dump(classifier, f)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()