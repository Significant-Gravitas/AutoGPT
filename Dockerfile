FROM python:3.11-slim

WORKDIR /app

# Install only the coaching module dependencies
RUN pip install --no-cache-dir \
    "fastapi>=0.100.0" \
    "uvicorn[standard]>=0.22.0" \
    "anthropic>=0.20.0" \
    "supabase>=2.0.0" \
    "pydantic>=2.0.0" \
    "python-dotenv>=1.0.0" \
    "requests>=2.32.0" \
    "urllib3>=2.0.0" \
    "charset-normalizer>=3.0.0" \
    "python-telegram-bot>=20.0"

# Copy only what the coaching API needs
COPY autogpt/singleton.py ./autogpt/singleton.py
COPY autogpt/coaching/ ./autogpt/coaching/

# Write a minimal autogpt package init — does NOT load the full autogpt app
# (avoids OpenAI key checks from cached old images / system packages)
RUN printf 'from dotenv import load_dotenv\nload_dotenv(override=True)\n' \
    > ./autogpt/__init__.py

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn autogpt.coaching.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
