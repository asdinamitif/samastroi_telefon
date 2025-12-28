FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Минимальные системные зависимости (без браузера)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код и данные
COPY samastroi_scraper.py /app/samastroi_scraper.py
COPY samastroi_extensions.py /app/samastroi_extensions.py
COPY onzs_catalog.xlsx /app/onzs_catalog.xlsx
COPY data /app/data

CMD ["python", "samastroi_scraper.py"]
