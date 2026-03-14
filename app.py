# import atexit
# import functools
# from queue import Queue
# from threading import Event, Thread

# from paddleocr import PaddleOCR, draw_ocr
# from PIL import Image
# import gradio as gr
# from pdf2image import convert_from_path
# import os
# import numpy as np


# LANG_CONFIG = {
#     "ch": {"num_workers": 2},
#     "en": {"num_workers": 2},
#     "fr": {"num_workers": 1},
#     "german": {"num_workers": 1},
#     "korean": {"num_workers": 1},
#     "japan": {"num_workers": 1},
# }
# CONCURRENCY_LIMIT = 8

# class PaddleOCRModelManager(object):
#     def __init__(self,
#                  num_workers,
#                  model_factory):
#         super().__init__()
#         self._model_factory = model_factory
#         self._queue = Queue()
#         self._workers = []
#         self._model_initialized_event = Event()
#         for _ in range(num_workers):
#             worker = Thread(target=self._worker, daemon=False)
#             worker.start()
#             self._model_initialized_event.wait()
#             self._model_initialized_event.clear()
#             self._workers.append(worker)

#     def infer(self, *args, **kwargs):
#         # XXX: Should I use a more lightweight data structure, say, a future?
#         result_queue = Queue(maxsize=1)
#         self._queue.put((args, kwargs, result_queue))
#         success, payload = result_queue.get()
#         if success:
#             return payload
#         else:
#             raise payload

#     def close(self):
#         for _ in self._workers:
#             self._queue.put(None)
#         for worker in self._workers:
#             worker.join()

#     def _worker(self):
#         model = self._model_factory()
#         self._model_initialized_event.set()
#         while True:
#             item = self._queue.get()
#             if item is None:
#                 break
#             args, kwargs, result_queue = item
#             try:
#                 result = model.ocr(*args, **kwargs)
#                 result_queue.put((True, result))
#             except Exception as e:
#                 result_queue.put((False, e))
#             finally:
#                 self._queue.task_done()


# def create_model(lang):
#     return PaddleOCR(lang=lang, use_angle_cls=True, use_gpu=False)


# model_managers = {}
# for lang, config in LANG_CONFIG.items():
#     model_manager = PaddleOCRModelManager(config["num_workers"], functools.partial(create_model, lang=lang))
#     model_managers[lang] = model_manager


# def close_model_managers():
#     for manager in model_managers.values():
#         manager.close()


# # XXX: Not sure if gradio allows adding custom teardown logic
# atexit.register(close_model_managers)


# # def inference(img, lang):
# #     ocr = model_managers[lang]
# #     result = ocr.infer(img, cls=True)[0]
# #     img_path = img
# #     image = Image.open(img_path).convert("RGB")
# #     boxes = [line[0] for line in result]
# #     txts = [line[1][0] for line in result]
# #     scores = [line[1][1] for line in result]
# #     im_show = draw_ocr(image, boxes, txts, scores,
# #                     font_path="./simfang.ttf")
# #     return im_show

# # def inference(img, lang):
# #     ocr = model_managers[lang]
# #     result = ocr.infer(img, cls=True)[0]

# #     extracted_text = []
# #     for line in result:
# #         text = line[1][0]
# #         confidence = line[1][1]
# #         extracted_text.append(f"{text} ({confidence:.2f})")

# #     return "\n".join(extracted_text)

# def inference(file_path, lang):
#     ocr = model_managers[lang]
#     extracted_text = []

#     # CASE 1: PDF input
#     if file_path.lower().endswith(".pdf"):
#         pages = convert_from_path(file_path, dpi=300)

#         for page_num, page in enumerate(pages, start=1):
#             # result = ocr.infer(page, cls=True)[0]
#             extracted_text.append(f"\n--- Page {page_num} ---\n")
#             page_np = np.array(page)
#             result = ocr.infer(page_np, cls=True)[0]
            
#             for line in result:
#                 text = line[1][0]
#                 confidence = line[1][1]
#                 extracted_text.append(f"{text} ({confidence:.2f})")

#     # CASE 2: Image input
#     else:
#         result = ocr.infer(file_path, cls=True)[0]
#         for line in result:
#             text = line[1][0]
#             confidence = line[1][1]
#             extracted_text.append(f"{text} ({confidence:.2f})")

#     return "\n".join(extracted_text)




# title = 'PaddleOCR'
# description = '''
# - Gradio demo for PaddleOCR. PaddleOCR demo supports Chinese, English, French, German, Korean and Japanese. 
# - To use it, simply upload your image and choose a language from the dropdown menu, or click one of the examples to load them. Read more at the links below.
# - [Docs](https://paddlepaddle.github.io/PaddleOCR/), [Github Repository](https://github.com/PaddlePaddle/PaddleOCR).
# '''

# examples = [
#     ['en_example.jpg','en'],
#     ['cn_example.jpg','ch'],
#     ['jp_example.jpg','japan'],
# ]

# css = ".output_image, .input_image {height: 40rem !important; width: 100% !important;}"
# # gr.Interface(
# #     inference,
# #     [
# #         gr.Image(type='filepath', label='Input'),
# #         gr.Dropdown(choices=list(LANG_CONFIG.keys()), value='en', label='language')
# #     ],
# #     # gr.Image(type='pil', label='Output'),
# #     gr.Textbox(label="OCR Extracted Text", lines=20),
# #     title=title,
# #     description=description,
# #     examples=examples,
# #     cache_examples=False,
# #     css=css,
# #     concurrency_limit=CONCURRENCY_LIMIT,
# #     ).launch(debug=False)

# gr.Interface(
#     inference,
#     [
#         gr.File(label="Upload Image or PDF"),
#         gr.Dropdown(choices=list(LANG_CONFIG.keys()), value='en', label='language')
#     ],
#     gr.Textbox(label="OCR Extracted Text", lines=25),
#     title=title,
#     description=description,
#     examples=None,
#     cache_examples=None,
#     concurrency_limit=CONCURRENCY_LIMIT,
# ).launch(debug=False)



import atexit
import functools
from queue import Queue
from threading import Event, Thread

from paddleocr import PaddleOCR
from PIL import Image
import gradio as gr
import numpy as np

# --- Configuration ---
LANG_CONFIG = {
    "en": {"num_workers": 2},
    "ch": {"num_workers": 2},
    "fr": {"num_workers": 1},
    "german": {"num_workers": 1},
    "korean": {"num_workers": 1},
    "japan": {"num_workers": 1},
}
CONCURRENCY_LIMIT = 8

class PaddleOCRModelManager(object):
    def __init__(self, num_workers, model_factory):
        self._model_factory = model_factory
        self._queue = Queue()
        self._workers = []
        self._model_initialized_event = Event()
        for _ in range(num_workers):
            worker = Thread(target=self._worker, daemon=True)
            worker.start()
            self._model_initialized_event.wait()
            self._model_initialized_event.clear()
            self._workers.append(worker)

    def infer(self, *args, **kwargs):
        result_queue = Queue(maxsize=1)
        self._queue.put((args, kwargs, result_queue))
        success, payload = result_queue.get()
        if success: return payload
        else: raise payload

    def _worker(self):
        model = self._model_factory()
        self._model_initialized_event.set()
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
    return PaddleOCR(lang=lang, use_angle_cls=True, use_gpu=False)

model_managers = {lang: PaddleOCRModelManager(cfg["num_workers"], functools.partial(create_model, lang=lang)) 
                  for lang, cfg in LANG_CONFIG.items()}

atexit.register(lambda: [m.close() for m in model_managers.values()])

def group_text_by_line(results):
    if not results or not results[0]: return []
    res = results[0]
    def get_y_center(item): return sum(p[1] for p in item[0]) / 4
    sorted_boxes = sorted(res, key=get_y_center)
    lines, current_line = [], []
    for item in sorted_boxes:
        if not current_line:
            current_line.append(item)
            continue
        last_y = get_y_center(current_line[-1])
        last_h = abs(current_line[-1][0][2][1] - current_line[-1][0][0][1])
        if abs(get_y_center(item) - last_y) < (last_h * 0.5):
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
    if current_line: lines.append(current_line)
    
    formatted = []
    for line in lines:
        line.sort(key=lambda x: x[0][0][0])
        formatted.append(" ".join([f"{item[1][0]}" for item in line]))
    return formatted

# --- Updated Inference to Support Appending ---
def inference(editor_data, lang, existing_text):
    if editor_data is None or editor_data["composite"] is None:
        return existing_text
    
    image = editor_data["composite"]
    image_np = np.array(image.convert("RGB"))
    
    ocr = model_managers[lang]
    result = ocr.infer(image_np, cls=True)
    
    new_lines = group_text_by_line(result)
    new_text = "\n".join(new_lines) if new_lines else "[No text detected in this selection]"
    
    # Combine with previous results
    separator = "\n" + "-"*30 + "\n"
    if existing_text.strip():
        combined_text = existing_text + separator + new_text
    else:
        combined_text = new_text
        
    return combined_text

def clear_results():
    return ""

# --- UI Layout ---
with gr.Blocks(css="textarea { font-family: monospace; }") as demo:
    gr.Markdown("## PaddleOCR - Multi-Crop Accumulator")
    
    with gr.Row():
        with gr.Column(scale=2):
            input_img = gr.ImageEditor(
                label="Step 1: Crop Area -> Step 2: Apply -> Step 3: Submit", 
                type="pil", 
                interactive=True,
                transforms=["crop"] 
            )
            with gr.Row():
                lang_drop = gr.Dropdown(choices=list(LANG_CONFIG.keys()), value='en', label='Language')
                submit_btn = gr.Button("Extract & Append", variant="primary")
                clear_btn = gr.Button("Clear All Text")

        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Accumulated OCR Results", lines=25, show_copy_button=True)

    # Logic for button clicks
    submit_btn.click(
        fn=inference, 
        inputs=[input_img, lang_drop, output_text], 
        outputs=output_text
    )
    
    clear_btn.click(
        fn=clear_results,
        outputs=output_text
    )

demo.launch()





# import atexit
# import functools
# from queue import Queue
# from threading import Event, Thread

# from paddleocr import PaddleOCR
# from PIL import Image
# import gradio as gr
# import numpy as np

# # --- Configuration ---
# LANG_CONFIG = {
#     "en": {"num_workers": 2},
#     "ch": {"num_workers": 2},
#     "fr": {"num_workers": 1},
#     "german": {"num_workers": 1},
#     "korean": {"num_workers": 1},
#     "japan": {"num_workers": 1},
# }
# CONCURRENCY_LIMIT = 8

# class PaddleOCRModelManager(object):
#     def __init__(self, num_workers, model_factory):
#         self._model_factory = model_factory
#         self._queue = Queue()
#         self._workers = []
#         self._model_initialized_event = Event()
#         for _ in range(num_workers):
#             worker = Thread(target=self._worker, daemon=True)
#             worker.start()
#             self._model_initialized_event.wait()
#             self._model_initialized_event.clear()
#             self._workers.append(worker)

#     def infer(self, *args, **kwargs):
#         result_queue = Queue(maxsize=1)
#         self._queue.put((args, kwargs, result_queue))
#         success, payload = result_queue.get()
#         if success: return payload
#         else: raise payload

#     def _worker(self):
#         model = self._model_factory()
#         self._model_initialized_event.set()
#         while True:
#             item = self._queue.get()
#             if item is None: break
#             args, kwargs, result_queue = item
#             try:
#                 result = model.ocr(*args, **kwargs)
#                 result_queue.put((True, result))
#             except Exception as e:
#                 result_queue.put((False, e))
        
#     def close(self):
#         for _ in self._workers: self._queue.put(None)
#         for worker in self._workers: worker.join()

# def create_model(lang):
#     # det_db_unclip_ratio: helps capture thinner, colored fonts (like the blue SRI text)
#     # use_angle_cls: helps with vertical or tilted text in scan margins
#     return PaddleOCR(lang=lang, use_angle_cls=True, use_gpu=False, det_db_unclip_ratio=2.0)

# model_managers = {lang: PaddleOCRModelManager(cfg["num_workers"], functools.partial(create_model, lang=lang)) 
#                   for lang, cfg in LANG_CONFIG.items()}

# atexit.register(lambda: [m.close() for m in model_managers.values()])

# def group_text_by_line(results):
#     """
#     Groups OCR results into horizontal lines and inserts proportional spacing
#     based on the physical gaps between text boxes in the scan image.
#     """
#     if not results or not results[0]: return []
#     res = results[0]
    
#     # Calculate vertical center of a box
#     def get_y_center(item):
#         box = item[0]
#         return (box[0][1] + box[2][1]) / 2

#     # Sort all detections by their vertical position
#     sorted_res = sorted(res, key=lambda x: get_y_center(x))
    
#     lines, current_line = [], []
#     if not sorted_res: return []
    
#     # Vertical grouping
#     current_line = [sorted_res[0]]
#     for i in range(1, len(sorted_res)):
#         last_item = current_line[-1]
#         curr_item = sorted_res[i]
#         last_y = get_y_center(last_item)
#         curr_y = get_y_center(curr_item)
#         h = abs(last_item[0][2][1] - last_item[0][0][1])
        
#         # Group if vertical centers are within 60% of text height
#         if abs(curr_y - last_y) < h * 0.6: 
#             current_line.append(curr_item)
#         else:
#             lines.append(current_line)
#             current_line = [curr_item]
#     lines.append(current_line)
    
#     # Process each line for horizontal spacing
#     output = []
#     for line in lines:
#         line.sort(key=lambda x: x[0][0][0]) # Sort Left-to-Right
        
#         line_str = ""
#         for i in range(len(line)):
#             text = line[i][1][0]
#             confidence = line[i][1][1]
#             if confidence < 0.35: continue # Filter low-confidence noise

#             if i > 0:
#                 # Calculate the visual gap between boxes
#                 prev_box_end_x = line[i-1][0][1][0] 
#                 curr_box_start_x = line[i][0][0][0]
#                 visual_gap = curr_box_start_x - prev_box_end_x
                
#                 # Estimate char width to determine how many spaces to insert
#                 box_width = line[i][0][1][0] - line[i][0][0][0]
#                 char_width = box_width / max(len(text), 1)
                
#                 if char_width > 0:
#                     num_spaces = int(visual_gap / char_width)
#                     line_str += " " * max(1, min(num_spaces, 12)) # Cap spaces at 12
#                 else:
#                     line_str += " "
            
#             line_str += text
            
#         if line_str.strip():
#             output.append(line_str)
#     return output

# def inference(editor_data, lang, existing_text):
#     if editor_data is None or editor_data["composite"] is None:
#         return existing_text
    
#     image = editor_data["composite"]
#     image_np = np.array(image.convert("RGB"))
    
#     ocr = model_managers[lang]
#     result = ocr.infer(image_np, cls=True)
    
#     new_lines = group_text_by_line(result)
#     new_text = "\n".join(new_lines) if new_lines else "[No text detected]"
    
#     session_divider = "\n" + "-"*20 + " NEW SCAN AREA " + "-"*20 + "\n"
#     if existing_text.strip():
#         combined_text = existing_text + session_divider + new_text
#     else:
#         combined_text = new_text
        
#     return combined_text

# # --- UI ---
# with gr.Blocks(css="textarea { font-family: 'Courier New', monospace; font-size: 13px; }") as demo:
#     gr.Markdown("### 🏥 Ultrasound Metric Extractor")
#     with gr.Row():
#         with gr.Column(scale=5):
#             input_img = gr.ImageEditor(label="Scan Image", type="pil", transforms=["crop"])
#             submit_btn = gr.Button("Extract Selected Data", variant="primary")
#         with gr.Column(scale=4):
#             output_text = gr.Textbox(label="Formatted Output", lines=25, show_copy_button=True)

#     submit_btn.click(fn=inference, inputs=[input_img, gr.State('en'), output_text], outputs=output_text)

# demo.launch()

