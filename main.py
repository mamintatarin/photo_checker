import os
import base64
import argparse
import json
import cv2
import requests
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Конфигурация
OLLAMA_URL = "http://localhost:11434/api/generate"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Создаем папку для загрузки, если она не существует
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Проверяет, имеет ли файл разрешенное расширение"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_face_opencv(image_path):
    """
    Проверяет наличие одного реального человека и четкость лица на изображении
    с помощью OpenCV
    """
    # Загружаем изображение
    img = cv2.imread(image_path)

    # Преобразуем в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Загружаем предобученный каскад Хаара для обнаружения лиц
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Обнаруживаем лица
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Используем HOG детектор для обнаружения людей
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Обнаруживаем людей на изображении
    boxes, _ = hog.detectMultiScale(
        gray, winStride=(8, 8), padding=(32, 32), scale=1.05
    )

    # Проверяем, есть ли хотя бы одно лицо и один человек
    faces_count = len(faces)
    people_count = len(boxes)

    if faces_count >= 1 and people_count >= 1:
        return (
            True,
            f"One or more faces and people detected (faces: {faces_count}, people: {people_count})",
        )

    return (
        False,
        f"No faces or people detected (faces: {faces_count}, people: {people_count})",
    )


def check_resp_str(resp: str) -> None | dict:
    """
    Проверяем формат ответа и выдаем словарь, если всё хорошо
    """

    # Просто пытаемся распарсить JSON, без сложных преобразований
    try:
        parsed_result = json.loads(resp)
    except json.JSONDecodeError:
        if len(resp) > 2:
            try:
                parsed_result = json.loads(resp[1:-1])
            except json.JSONDecodeError:
                return None
        else:
            return None

    if not isinstance(parsed_result, dict):
        return None

    # Проверяем обязательные поля
    if (
        "match" in parsed_result
        and "appearance_score" in parsed_result
        and "single_clear_person" in parsed_result
        and "gender" in parsed_result
    ):
        # Убедимся, что значения имеют правильный тип
        if not isinstance(parsed_result["match"], bool):
            return None

        if not isinstance(parsed_result["appearance_score"], (int, float)):
            return None
        else:
            parsed_result["appearance_score"] = float(parsed_result["appearance_score"])

        if not isinstance(parsed_result["single_clear_person"], bool):
            return None

        # Обработка поля gender
        if parsed_result["gender"] not in {"male", "female"}:
            return None

        return parsed_result

    return None


def query_ollama(image_path, text_description, attempts: int = 3) -> dict | None:
    """
    Отправляет изображение и текст в Ollama и получает результат
    """
    # Подготовляем запрос к Ollama
    with open(image_path, "rb") as img_file:
        # Кодируем изображение в base64 для передачи в JSON
        

        image_data = base64.b64encode(img_file.read()).decode("utf-8")

    # Более понятный промпт для модели
    prompt = f"""
    Проанализируйте изображение и сравните его с текстовым описанием: '{text_description}'
    
    Ответьте строго в формате JSON без дополнительного текста:
    {{
        "match": true/false - соответствует ли описание фото,
        "appearance_score": число от 0 до 10 - оценка внешности,
        "single_clear_person": true/false - на фото только один человек с четко видимым лицом,
        "gender": "male"/"female"/null - пол человека на фото
    }}
    """

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_data],  # Передаем изображение в base64
        "stream": False,
        "options": {
            "temperature": 0  # Отключаем "мышление" модели
        },
    }

    # Отправляем запрос
    response = None
    for attempt in range(attempts):  # Две попытки
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=200)
            if response.status_code == 200:
                try:
                    # Парсим JSON ответ
                    result = response.json()
                    if "response" not in result or not isinstance(result, dict):
                        raise ValueError("Invalid response format from Ollama")
                    # Проверяем формат результата
                    result_parsed = check_resp_str(result["response"])
                    if result_parsed:
                        return result_parsed
                    raise ValueError("Формат ответа не совпал с ожидаемым")
                except (json.JSONDecodeError, ValueError) as e:
                    # Просто повторяем попытку, если не удалось распознать JSON
                    if attempt != 0:
                        continue
                    else:
                        raise e
        except requests.exceptions.RequestException as e:
            if attempt == 0:  # Если последняя попытка
                raise e
    return None


@app.route("/")
def index():
    """Отображает веб-интерфейс для загрузки изображения и ввода текста"""
    template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Photo Checker</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 600px; margin: 0 auto; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input[type="file"], textarea { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; }
            textarea { height: 100px; resize: vertical; }
            button { background-color: #4CAF50; color: white; padding: 12px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background-color: #45a049; }
            .result { margin-top: 20px; padding: 15px; background-color: #f0f0f0; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Photo Checker</h1>
            <form method="POST" action="/analyze" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="image">Выберите изображение:</label>
                    <input type="file" id="image" name="image" accept=".png,.jpg,.jpeg" required>
                </div>
                <div class="form-group">
                    <label for="description">Описание человека на фото:</label>
                    <textarea id="description" name="description" placeholder="Введите описание человека на фото..." required></textarea>
                </div>
                <button type="submit">Анализировать</button>
            </form>
        </div>
    </body>
    </html>
    """
    return render_template_string(template)


@app.route("/analyze", methods=["POST"])
def analyze():
    """Обрабатывает загруженное изображение и текст, возвращает результат анализа"""
    try:
        # Проверяем, был ли загружен файл
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]

        # Проверяем, что файл выбран
        if file.filename == "":
            return jsonify({"error": "No image selected"}), 400

        # Проверяем, что файл имеет разрешенное расширение
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
        else:
            return jsonify({"error": "Invalid file type"}), 400

        # Получаем текстовое описание
        text_description = request.form.get("description", "")

        if not text_description:
            return jsonify({"error": "Description is required"}), 400

        # Проверяем изображение с помощью OpenCV, если не указан параметр пропуска
        if not SKIP_OPENCV_CHECK:
            face_detected, face_msg = detect_face_opencv(filepath)

            if not face_detected:
                return jsonify(
                    {
                        "success": False,
                        "message": face_msg,
                        "match": False,
                        "appearance_score": 0,
                        "single_clear_person": False,
                    }
                )

        # Отправляем изображение и текст в Ollama
        result = query_ollama(filepath, text_description, attempts=ATTEMPTS)

        # Возвращаем результат
        return jsonify(
            {
                "success": True,
                "message": "Analysis completed successfully",
                "match": result["match"],
                "appearance_score": result["appearance_score"],
                "single_clear_person": result["single_clear_person"],
                "gender": result["gender"],
            }
        )

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500
    except json.JSONDecodeError as e:
        return jsonify({"error": str(e)}), 500
    except ValueError as e:
        return jsonify({"error": str(e)}), 500


# Глобальная переменная для хранения настроек
SKIP_OPENCV_CHECK = False
ATTEMPTS = 3
MODEL_NAME = "qwen3-vl:2b"  # Default model


def main():
    "главная функция"
    parser = argparse.ArgumentParser(description="Photo Checker Service")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to listen on (default: 5000)"
    )
    parser.add_argument(
        "--model", type=str, default="qwen3-vl:2b", help="Model name to use (default: qwen3-vl:2b)"
    )
    parser.add_argument(
        "--skip-opencv-check",
        action="store_true",
        help="Skip OpenCV face detection check",
    )
    parser.add_argument(
        "--attempts", default=3, help="Number of attempts to question ollama model"
    )

    args = parser.parse_args()

    # Устанавливаем глобальные переменные
    global SKIP_OPENCV_CHECK
    global ATTEMPTS
    global MODEL_NAME
    SKIP_OPENCV_CHECK = args.skip_opencv_check
    ATTEMPTS = args.attempts
    MODEL_NAME = args.model

    print(f"Starting Photo Checker Service on {args.host}:{args.port}")
    print(f"Using model: {MODEL_NAME}")
    print(
        f"OpenCV face detection check: {'disabled' if SKIP_OPENCV_CHECK else 'enabled'}"
    )
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
