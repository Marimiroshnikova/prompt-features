FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8765 \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken

WORKDIR /app

# Dependencies first, so code edits do not invalidate the layer that installs
# the 400 MB of NLP wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY . .

EXPOSE 8765
CMD ["python", "app.py"]
