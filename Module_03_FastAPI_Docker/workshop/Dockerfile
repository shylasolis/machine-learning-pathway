FROM python:3.14-slim
WORKDIR /workshop
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY artifacts/ artifacts/
EXPOSE 8000
USER 10001
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
