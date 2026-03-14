import atexit
import functools
import shutil
import os
import uvicorn
import numpy as np
import cv2
import json
from queue import Queue
from threading import Event, Thread
from typing import List
import re

import gradio as gr
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw

# --- Configuration ---
LANG_CONFIG = {
    "en": {"num_workers": 2},
}

# --- Image Preprocessing ---
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for better OCR accuracy: Grayscale, Resizing, Binarization."""
    img_np = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # 1. Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Resizing (Scale up by 1.5x)
    width = int(gray.shape[1] * 1.5)
    height = int(gray.shape[0] * 1.5)
    dim = (width, height)
    resized = cv2.resize(gray, dim, interpolation=cv2.INTER_CUBIC)
    
    # 3. Denoising & Adaptive Binarization
    blur = cv2.GaussianBlur(resized, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
    )
    
    # Convert back to 3 channel RGB for PaddleOCR
    processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    return processed

# --- Structured Data Extraction ---
def extract_bank_form_data(lines: List[str]) -> dict:
    """Heuristic parser to convert text lines to Bank Form Structured Data."""
    data = {
        "Name": {"First": "", "Last": "", "Given Name": ""},
        "Date of Birth": "",
        "Residential Address": {
            "Street Address": "",
            "Street Address Line 2": "",
            "City": "",
            "Region": "",
            "Postal/Zip Code": "",
            "Country": ""
        },
        "Contact": {"Phone": "", "Email": ""},
        "Account Details": {
            "Bank Name": "",
            "Branch": "",
            "Address": "",
            "Account Name": "",
            "Account Type": ""
        }
    }
    
    full_text = " \n ".join(lines)
    
    # Email matching
    email_match = re.search(r"[\w\.-]+@[\w\.-]+", full_text)
    if email_match:
        data["Contact"]["Email"] = email_match.group(0)
        
    # Phone matching
    phone_match = re.search(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", full_text)
    if phone_match:
        data["Contact"]["Phone"] = phone_match.group(0)
        
    # DOB matching
    dob_match = re.search(r"\d{2}/\d{2}/\d{4}", full_text)
    if dob_match:
        data["Date of Birth"] = dob_match.group(0)
        
    # Heuristics based on lines
    for i, line in enumerate(lines):
        lower_line = line.strip().lower()
        
        if "name of bank" in lower_line or "credit union" in lower_line:
            if i + 1 < len(lines): data["Account Details"]["Bank Name"] = lines[i+1]
        elif "branch" in lower_line and len(lower_line) < 10:
            if i + 1 < len(lines): data["Account Details"]["Branch"] = lines[i+1]
        elif "account type" in lower_line:
            if i + 1 < len(lines): data["Account Details"]["Account Type"] = lines[i+1]
        elif "account in the name" in lower_line:
            if i + 1 < len(lines): data["Account Details"]["Account Name"] = lines[i+1]
            
    return data

# --- OCR Manager ---
class PaddleOCRModelManager:
    def __init__(self, num_workers, model_factory):
        self._model_factory = model_factory
        self._queue = Queue()
        self._workers = []
        for _ in range(num_workers):
            init_event = Event()
            worker = Thread(target=self._worker, args=(init_event,), daemon=True)
            worker.start()
            init_event.wait()
            self._workers.append(worker)

    def infer(self, *args, **kwargs):
        result_queue = Queue(maxsize=1)
        self._queue.put((args, kwargs, result_queue))
        success, payload = result_queue.get()
        if success: return payload
        else: raise payload

    def _worker(self, init_event):
        try:
            model = self._model_factory()
        finally:
            init_event.set()
        while True:
            item = self._queue.get()
            if item is None: break
            args, kwargs, result_queue = item
            try:
                result = model.ocr(*args, **kwargs)
                result_queue.put((True, result))
            except Exception as e:
                result_queue.put((False, e))
        
    def close(self):
        for _ in self._workers: self._queue.put(None)
        for worker in self._workers: worker.join()

def create_model(lang):
    return PaddleOCR(lang=lang, use_angle_cls=True, use_gpu=False, show_log=False)

# Initialize Managers
model_managers = {lang: PaddleOCRModelManager(cfg["num_workers"], functools.partial(create_model, lang=lang)) 
                  for lang, cfg in LANG_CONFIG.items()}

atexit.register(lambda: [m.close() for m in model_managers.values()])

# --- Text Formatting ---
def group_text_by_line(results):
    if not results or not results[0]: return []
    res = results[0]
    
    def get_metrics(item):
        points = item[0]
        y_center = sum(p[1] for p in points) / 4
        height = abs(points[2][1] - points[0][1])
        return y_center, height

    sorted_boxes = sorted(res, key=lambda x: get_metrics(x)[0])
    lines, current_line = [], []
    
    for item in sorted_boxes:
        if not current_line:
            current_line.append(item)
            continue
        curr_y, _ = get_metrics(item)
        last_y, last_h = get_metrics(current_line[-1])
        if abs(curr_y - last_y) < (last_h * 0.5):
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
    
    if current_line: lines.append(current_line)
    
    formatted = []
    for line in lines:
        line.sort(key=lambda x: x[0][0][0])
        formatted.append(" ".join([str(item[1][0]) for item in line]))
    return formatted

# --- Main Inference Logic ---
def inference(file_paths, lang="en", apply_preprocessing=True):
    if not file_paths:
        return "No files uploaded.", [], None, "0/0", 0
    
    ocr_manager = model_managers[lang]
    all_results_text = ""
    processed_images_list = [] 
    
    for i, file_path in enumerate(file_paths):
        try:
            full_image = Image.open(file_path).convert("RGB")
            filename = os.path.basename(file_path)
            display_image = full_image.copy()
            
            # Application of Preprocessing Pipeline
            if apply_preprocessing:
                img_to_ocr = preprocess_image(full_image)
            else:
                img_to_ocr = np.array(full_image)
            
            # OCR Inference
            result = ocr_manager.infer(img_to_ocr, cls=True)
            lines = group_text_by_line(result)
            
            # Structured Processing
            structured_data = extract_bank_form_data(lines)
            
            separator = "="*40
            file_header = f"\n{separator}\n FILE {i+1}: {filename}\n{separator}\n"
            
            # Pretty print JSON structure
            json_output = json.dumps(structured_data, indent=4)
            
            all_results_text += file_header + "\n=== Structured Output ===\n" + json_output + "\n\n=== Raw Extracted Lines ===\n" + "\n".join(lines) + "\n"
            processed_images_list.append(display_image)

        except Exception as e:
            all_results_text += f"\n[Error processing file {i+1}]: {str(e)}\n"

    first_image = processed_images_list[0] if processed_images_list else None
    count_str = f"Image 1/{len(processed_images_list)}" if processed_images_list else "0/0"
    return all_results_text, processed_images_list, first_image, count_str, 0

# --- Gradio UI Setup ---
def navigate_images(images, current_index, direction):
    if not images: return None, "0/0", 0
    new_index = (current_index + direction) % len(images)
    return images[new_index], f"Image {new_index + 1}/{len(images)}", new_index

with gr.Blocks(title="Bank Form Extraction Engine") as demo:
    gr.Markdown("## 🏦 Bank Form Extraction Engine")
    gr.Markdown("Upload bank forms. The engine will preprocess the images and extract vital entities like Name, DOB, and Account Details into a structured JSON format.")
    images_state = gr.State([])
    index_state = gr.State(0)
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload Bank Forms", file_count="multiple", type="filepath")
            apply_preprocess = gr.Checkbox(label="Apply Image Preprocessing (Enhances accuracy for poor scans)", value=True)
            lang_drop = gr.Dropdown(choices=list(LANG_CONFIG.keys()), value='en', label='Language', visible=False)
            submit_btn = gr.Button("🚀 Extract Form Data", variant="primary")
            output_image = gr.Image(label="Uploaded Image", interactive=False)
            with gr.Row():
                prev_btn = gr.Button("◀ Previous")
                counter_label = gr.Label(value="0/0", show_label=False)
                next_btn = gr.Button("Next ▶")
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Extracted Structured Data", lines=30)

    submit_btn.click(inference, [file_input, lang_drop, apply_preprocess], [output_text, images_state, output_image, counter_label, index_state])
    next_btn.click(functools.partial(navigate_images, direction=1), [images_state, index_state], [output_image, counter_label, index_state])
    prev_btn.click(functools.partial(navigate_images, direction=-1), [images_state, index_state], [output_image, counter_label, index_state])

# --- FastAPI Integration ---
app = FastAPI(title="Bank Form Extraction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/ocr")
async def api_ocr(
    file: UploadFile = File(...), 
    apply_preprocessing: bool = Form(True),
    lang: str = Form("en")
):
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # We pass a list [file_path] as the inference function handles multiple files
        text, _, _, count_info, _ = inference([file_path], lang, apply_preprocessing)
        
        # We need to extract just the JSON part for the API response
        try:
            json_str = text.split("=== Structured Output ===")[1].split("=== Raw Extracted Lines ===")[0].strip()
            parsed_json = json.loads(json_str)
        except:
            parsed_json = {}
            
        return {
            "status": "success",
            "filename": file.filename,
            "extracted_data": parsed_json,
            "raw_text_output": text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# Mount Gradio UI to FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)