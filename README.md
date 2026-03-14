# Team Mavericks – PaddleOCR Multi-Crop Accumulator

## Overview

This project is a **multi-language OCR (Optical Character Recognition) web application** built with [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) and [Gradio](https://gradio.app/). It lets users upload an image, draw crop selections over specific regions, and extract text from those regions — accumulating results across multiple crops in a single session.

## Features

- **Multi-language support**: English, Chinese, French, German, Korean, Japanese
- **Interactive crop tool**: Draw a selection box on any part of the uploaded image to extract only the text in that region
- **Accumulated results**: Each extraction appends to a running text output, separated by dividers — useful for scanning multiple sections of a document
- **Angle classification**: Handles tilted or rotated text automatically
- **Copy-to-clipboard**: The output textbox includes a one-click copy button
- **Concurrent model workers**: Each language has dedicated worker threads to handle parallel inference requests efficiently

## Tech Stack

| Component | Library |
|-----------|---------|
| OCR Engine | `paddleocr`, `paddlepaddle` |
| Web UI | `gradio` |
| Image Processing | `Pillow`, `opencv-python`, `numpy` |
| PDF Support | `pdf2image`, `poppler-utils` |
| Containerization | Docker (base image: `registry.hf.space/paddlepaddle-paddleocr:latest`) |

## How It Works

1. **Upload an image** using the interactive image editor.
2. **Crop a region** of interest with the built-in crop tool.
3. **Select a language** from the dropdown.
4. **Click "Extract & Append"** — the OCR engine processes the cropped region and appends the detected text to the output panel.
5. Repeat steps 2–4 for additional regions; results accumulate with clear separators.
6. **Click "Clear All Text"** to reset the output and start fresh.

## Running Locally

### Prerequisites

- Python 3.8+
- `pip`

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the application

```bash
python app.py
```

The Gradio interface will launch and print a local URL (e.g. `http://127.0.0.1:7860`).

## Running with Docker

```bash
docker build -t team-mavericks-ocr .
docker run -p 7860:7860 team-mavericks-ocr
```

Then open `http://localhost:7860` in your browser.

## Project Structure

```
.
├── app.py              # Main application – Gradio UI + OCR inference logic
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition (based on official PaddleOCR image)
└── fastapi_version     # Historical FastAPI variant (reference files)
```

## Configuration

Language concurrency can be tuned in `app.py` via the `LANG_CONFIG` dictionary:

```python
LANG_CONFIG = {
    "en": {"num_workers": 2},
    "ch": {"num_workers": 2},
    "fr": {"num_workers": 1},
    ...
}
```

Increase `num_workers` for a language if you expect heavy concurrent traffic for that language. The overall `CONCURRENCY_LIMIT` controls how many simultaneous Gradio requests are served.
