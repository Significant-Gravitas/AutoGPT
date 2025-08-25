import os

# Set SUPABASE_JWT_SECRET, otherwise the config module fails to load
os.environ["SUPABASE_JWT_SECRET"] = "placeholder-secret-key-that-is-long-enough"
