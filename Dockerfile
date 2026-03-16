FROM python:3.11-slim

WORKDIR /app

# System deps for openpyxl/xlrd, pyarrow, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Parquet data lives on a mounted volume
ENV PER_TICKER_PARQUET_DIR=/data/parquet
ENV PER_TICKER_MINUTE_DIR=/data/minute

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app18.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
