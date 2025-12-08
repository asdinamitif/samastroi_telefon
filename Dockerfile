FROM python:3.11-slim

WORKDIR /app

# Копируем файлы
COPY . .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Пробрасываем переменные Railway внутрь контейнера
ENV TG_API_ID=${TG_API_ID}
ENV TG_API_HASH=${TG_API_HASH}
ENV SESSION_NAME=${SESSION_NAME}
ENV ADMIN_ID=${ADMIN_ID}
ENV YAGPT_API_KEY=${YAGPT_API_KEY}
ENV YAGPT_FOLDER_ID=${YAGPT_FOLDER_ID}
ENV LOG_LEVEL=${LOG_LEVEL}

CMD ["python", "samastroi_telethon.py"]
