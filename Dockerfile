FROM registry.hf.space/paddlepaddle-paddleocr:latest

# Install system deps for PDF
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --no-cache-dir pdf2image

# 🔥 COPY YOUR app.py INTO THE IMAGE
COPY app.py /home/user/app.py

# 🔥 FORCE CONTAINER TO RUN YOUR app.py
CMD ["python", "app.py"]




# # Using the official PaddlePaddle image for stability
# FROM registry.baidubce.com/paddlepaddle/paddle:2.5.2

# WORKDIR /app

# # Install system dependencies for OCR and image processing
# RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# # Copy requirements and install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the updated main.py
# COPY . .

# # Expose port 8000 (FastAPI and Gradio now share this)
# EXPOSE 8000

# # Start the application using Python (which triggers uvicorn.run)
# CMD ["python", "main.py"]