FROM python:3.11-slim

# Чтобы Python не создавал .pyc и всё писал сразу в stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Рабочая директория внутри контейнера
WORKDIR /app

# Ставим зависимости
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# На всякий случай создаём директории под данные/логи
RUN mkdir -p /app/data /app/logs

# Основные файлы бота
COPY samastroi_telethon.py /app/
COPY samastroi_telethon.session /app/

# Старт команды
CMD ["python", "samastroi_telethon.py"]
