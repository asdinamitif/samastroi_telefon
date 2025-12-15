FROM python:3.11-slim

WORKDIR /app

# Копируем requirements
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY samastroi_scraper.py .

# Переменная для volume
ENV DATA_DIR=/app/data

CMD ["python", "samastroi_scraper.py"]
