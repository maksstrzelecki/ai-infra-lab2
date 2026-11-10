FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 --retries 5 -r requirements.txt

COPY inference.py server.py load_balancer.py ./

EXPOSE 8000

CMD ["sh", "-c", "uvicorn server:app --host 127.0.0.1 --port 8001 & uvicorn server:app --host 127.0.0.1 --port 8002 & exec uvicorn load_balancer:app --host 0.0.0.0 --port 8000"]
