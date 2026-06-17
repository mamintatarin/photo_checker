# photo_checker
Automatic photo verification via text description and simple recognition models

## Installation

First, install ollama https://ollama.com/

Then, you need to download preferred model that supports images. Let's say it's qwen3-vl:2b, then

```
ollama pull qwen3-vl:2b
```
This command will download the model.

Next, you need to set up the Python environment:

1. Create a virtual environment:
```
python -m venv .venv
```

2. Activate the virtual environment:
- On Windows:
```
.venv\Scripts\activate
```
- On Linux/macOS:
```
source .venv/bin/activate
```

3. Install required packages:
```
pip install -r requirements.txt
```

Supported Python versions: 3.8 and higher


## Run

To start the server, run:
```
python main.py
```

Optional command line arguments:
- `--host`: Host address to bind to (default: 127.0.0.1)
- `--port`: Port to listen on (default: 5000)
- `--skip-opencv-check`: Skip OpenCV face detection check (default: disabled)

Example:
```
python main.py --host 0.0.0.0 --port 8080 --skip-opencv-check
```

When the server starts, it will show the address it's running on.
By default, it runs on http://127.0.0.1:5000

The OpenCV check verifies that there is at least one face and at least one person
detected on the image using Haar cascade for faces and HOG descriptor for people.
If the check is enabled (default), the image must pass both detections before being
sent to the Ollama model.

## Test

The web interface is available at the server address (by default: http://127.0.0.1:5000).
You can upload an image and enter a text description to test the functionality.

Additionally, you can use the test script `test_api.py` to test the API directly:

```
python test_api.py <image_path> <description>
```

Where:
- `<image_path>`: Path to the image file to upload
- `<description>`: Description of the person in the image

Example:
```
python test_api.py test_image.jpg "A man wearing glasses"
```

The script will send a request to the server and display the response with the analysis results.


