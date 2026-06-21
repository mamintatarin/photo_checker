import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis


def get_face_analysis_instance(providers=None):
    """
    Возвращает экземпляр FaceAnalysis с предустановленными параметрами
    """
    if providers is None:
        providers = ['CPUExecutionProvider']
    
    face_analysis = FaceAnalysis(name='buffalo_l', providers=providers)
    face_analysis.prepare(ctx_id=0, det_size=(640, 640))
    return face_analysis


def extract_face_embedding(image_path, face_analysis=None):
    """
    Извлечение вектора лица из изображения с помощью InsightFace
    """
    if face_analysis is None:
        face_analysis = get_face_analysis_instance()
    
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Detect faces
    faces = face_analysis.get(img)
    
    if not faces:
        return None
    
    # Return embedding of the first detected face
    return faces[0].embedding


def extract_face_embeddings(image_path, face_analysis=None):
    """
    Извлечение векторов всех лиц из изображения с помощью InsightFace
    """
    if face_analysis is None:
        face_analysis = get_face_analysis_instance()
    
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Detect faces
    faces = face_analysis.get(img)
    
    embeddings = []
    for face in faces:
        if hasattr(face, 'embedding'):
            embeddings.append(face.embedding)
    
    return embeddings