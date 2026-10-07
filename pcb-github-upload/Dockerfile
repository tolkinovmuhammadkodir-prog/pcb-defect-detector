FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Same relative layout as the repo, because service/app.py imports ../src/infer_onnx.py
COPY src/infer_onnx.py src/infer_onnx.py
COPY service/app.py service/app.py
COPY models/best.onnx models/best.onnx

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

ENV MODEL_PATH=/app/models/best.onnx
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
