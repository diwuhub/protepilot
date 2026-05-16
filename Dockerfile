FROM python:3.12-slim
WORKDIR /app
COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt
COPY src/ src/
COPY app.py .
COPY data/ data/
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
