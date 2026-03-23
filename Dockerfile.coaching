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
    "charset-normalizer>=3.0.0"

# Copy only the coaching module and singleton helper
COPY autogpt/__init__.py ./autogpt/__init__.py
COPY autogpt/singleton.py ./autogpt/singleton.py
COPY autogpt/coaching/ ./autogpt/coaching/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn autogpt.coaching.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
