FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=0
ENV CHROMIUM_FLAGS="--no-sandbox --disable-dev-shm-usage"

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
    tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m playwright install chromium

COPY samastroi_scraper.py .
COPY onzs_catalog.xlsx /app/onzs_catalog.xlsx
COPY data ./data

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "samastroi_scraper.py"]
