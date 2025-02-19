FROM python:3.10-slim

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    make \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \ 
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --index-url https://test.pypi.org/simple/ --no-deps inference_worker
RUN pip install pika requests python-dotenv fastapi[standard]
RUN pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy project files
COPY . .

# Set the command to run
CMD ["inference-worker", "start"]