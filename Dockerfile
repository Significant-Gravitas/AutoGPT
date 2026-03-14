FROM python:3.11-slim

WORKDIR /app

# Install coaching API dependencies from the dedicated requirements file.
# Adding a new package: edit requirements.coaching.txt — no Dockerfile changes needed.
COPY requirements.coaching.txt ./requirements.coaching.txt
RUN pip install --no-cache-dir -r requirements.coaching.txt

# Copy only what the coaching API needs (keeps image small)
COPY autogpt/singleton.py ./autogpt/singleton.py
COPY autogpt/coaching/ ./autogpt/coaching/

# Minimal autogpt package init — does NOT load the full AutoGPT app
# (prevents OpenAI key checks from old cached system packages)
RUN printf 'from dotenv import load_dotenv\nload_dotenv(override=True)\n' \
    > ./autogpt/__init__.py

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn autogpt.coaching.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
