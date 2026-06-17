import requests
import json
import argparse

def main():
    # Аргументы командной строки
    parser = argparse.ArgumentParser(description='Test the Photo Checker API')
    parser.add_argument('image_path', type=str, help='Path to the image file to upload')
    parser.add_argument('description', type=str, help='Description of the person in the image')
    
    args = parser.parse_args()
    
    # Тестирование API
    url = 'http://127.0.0.1:5000/analyze'

    # Отправляем POST-запрос с изображением и описанием
    files = {'image': open(args.image_path, 'rb')}
    data = {'description': args.description}

    try:
        print(f"Sending request with image: {args.image_path} and description: {args.description}")
        response = requests.post(url, files=files, data=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        # Если ответ успешный, попробуем распарсить JSON
        if response.status_code == 200:
            result = response.json()
            print("\nParsed Results:")
            print(f"- Success: {result.get('success')}")
            print(f"- Message: {result.get('message')}")
            print(f"- Match: {result.get('match')}")
            print(f"- Appearance Score: {result.get('appearance_score')}")
            print(f"- Single Clear Person: {result.get('single_clear_person')}")
            print(f"- Gender: {result.get('gender')}")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        files['image'].close()

if __name__ == '__main__':
    main()