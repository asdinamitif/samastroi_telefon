FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg \
    libnss3 libnspr4 \
    libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libasound2 \
    libpangocairo-1.0-0 libpango-1.0-0 libcairo2 \
    libgtk-3-0 \
    libx11-6 libx11-xcb1 libxcb1 libxext6 libxrender1 \
    libxshmfence1 libxss1 libxtst6 \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# ===== DIAGNOSTICS START =====
RUN echo "===== REQUIREMENTS.TXT USED BY RAILWAY ====="
RUN pwd
RUN ls -la
RUN cat requirements.txt
RUN echo "============================================"
# ===== DIAGNOSTICS END =====

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m playwright install chromium

COPY samastroi_scraper.py .
COPY onzs_catalog.xlsx /app/onzs_catalog.xlsx
COPY data ./data

CMD ["python", "samastroi_scraper.py"]
