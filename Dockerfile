FROM python:3.11-slim
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from insightface.app import FaceAnalysis; a=FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); a.prepare(ctx_id=-1)"
COPY main.py .
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
