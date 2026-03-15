# 🏦 Bank Form Extractor

An intelligent OCR pipeline for automatically extracting structured data from scanned bank application forms. It uses **PaddleOCR** for text extraction, **OpenCV** for targeted checkbox detection, and the **Groq LLM API** to parse the extracted text into clean, structured JSON.

---

# ✨ Features

| Feature                     | Description                                                                                          |
| --------------------------- | ---------------------------------------------------------------------------------------------------- |
| 📄 Multi-image Upload       | Process one or more scanned bank form images at once                                                 |
| 🔍 Full Text Extraction     | Extracts all text from the form exactly as it appears                                                |
| ☑️ Smart Checkbox Detection | Detects **only** the real checkboxes (Transaction Rights / View Only), ignoring character/date boxes |
| ✓ Tick Detection            | Checks if a checkbox has a handwritten tick/mark inside it                                           |
| 🤖 Groq JSON Parsing        | Sends extracted text to Groq LLM and returns structured JSON                                         |
| 🖼️ Visual Annotation       | Annotated preview image — 🟩 green = ticked, 🟥 red = empty                                          |
| 🌐 Gradio UI                | Clean, browser-based interface with image carousel                                                   |

---

# 🖥️ Demo

After running, visit:

```
http://localhost:7860
```

in your browser.

### Output Example

```
APPLICATION FORM FOR INTERNET BANKING (FOR INDIVIDUALS)
Transaction Rights   ✓ View Only
CUSTOMER ID: 123456189
ACCOUNT NO.: 16543203
NAME OF THE ACCOUNT HOLDER: RAJ
DATE OF BIRTH: 20-05-2000
```

### Structured JSON (via Groq)

```json
{
  "customer_id": "123456189",
  "account_number": "16543203",
  "account_holder_name": "RAJ",
  "date_of_birth": "20-05-2000",
  "transaction_rights": false,
  "view_only": true,
  "communication_address": {
    "city": "CHENNAI",
    "state": "TAMILNADU",
    "pin_code": "600126"
  }
}
```

---

# 🗂️ Project Structure

```
llm/
├── app.py
├── fastapi_version/
│   └── app_gradio.py
├── Dockerfile
├── requirements.txt
├── .env
└── README.md
```

| File                            | Description                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| `app.py`                        | Gradio app (crop-to-extract mode)                            |
| `fastapi_version/app_gradio.py` | Main application (multi-image upload + Groq JSON extraction) |
| `Dockerfile`                    | Docker container setup                                       |
| `requirements.txt`              | Python dependencies                                          |
| `.env`                          | API keys (not committed to GitHub)                           |

---

# 🚀 Quick Start

## Option 1 — Docker (Recommended)

### 1. Build the Docker image

```bash
docker build --platform=linux/amd64 -t pad2 .
```

### 2. Run the application

```powershell
docker run --rm -it `
  -p 7860:7860 `
  --platform=linux/amd64 `
  -v "${PWD}/fastapi_version/app_gradio.py:/app_gradio.py" `
  --env-file .env `
  pad2 python -u /app_gradio.py
```

Open in browser:

```
http://localhost:7860
```

---

## Option 2 — Local Python Setup

### 1. Install Dependencies

```bash
pip install paddlepaddle paddleocr gradio opencv-python groq python-dotenv pillow
```

### 2. Add Groq API key in `.env`

```
groqapi=your_groq_api_key_here
```

### 3. Run the Application

```bash
python fastapi_version/app_gradio.py
```

---

# ⚙️ How It Works

```
Scanned Form Image
       │
       ▼
  PaddleOCR
       │
       ▼
  Extract Text (Line-by-line)
       │
       ▼
  OpenCV Checkbox Detection
       │
       │  • Detect checkbox contours
       │  • Locate labels in OCR output
       │  • Search for square box near label
       │  • Measure ink density inside box
       │
       ▼
  Annotated Text Output
       │
       ▼
  Groq LLM (llama3-70b-8192)
       │
       ▼
  Structured JSON Output
```

---

# 🔧 Configuration

| Variable          | File            | Description                        |
| ----------------- | --------------- | ---------------------------------- |
| `groqapi`         | `.env`          | Your Groq API key                  |
| `LANG_CONFIG`     | `app_gradio.py` | OCR language configuration         |
| `CHECKBOX_LABELS` | `app_gradio.py` | Labels used for checkbox detection |

Example:

```
LANG_CONFIG = ['en', 'ch']
CHECKBOX_LABELS = ["Transaction Rights", "View Only"]
```

---

# 📦 Dependencies

| Package       | Purpose                      |
| ------------- | ---------------------------- |
| paddleocr     | OCR text extraction          |
| opencv-python | Checkbox contour detection   |
| gradio        | Web UI                       |
| groq          | LLM-powered JSON structuring |
| python-dotenv | Load environment variables   |
| Pillow        | Image processing             |

---

SAMPLE OUTPUT: output1.jpeg

# 🔐 Security Note

**Never commit your `.env` file to GitHub.**

Add this to `.gitignore`:

```
.env
```

---

# 📄 License

MIT License

Free to use, modify, and distribute.
