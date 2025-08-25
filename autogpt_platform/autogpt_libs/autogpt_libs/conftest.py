import os

# Set JWT_VERIFY_KEY, otherwise the config module fails to load
os.environ["JWT_VERIFY_KEY"] = "placeholder-secret-key-that-is-long-enough"
