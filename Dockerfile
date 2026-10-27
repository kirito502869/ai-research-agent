FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 依赖单独一层，改代码时不必重装依赖
COPY requirements.txt .
# 加长超时并重试：PyPI 在部分网络下会中途断流
RUN pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt

COPY . .

ENV OLLAMA_BASE_URL=http://ollama:11434

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
