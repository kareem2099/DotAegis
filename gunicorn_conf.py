import multiprocessing
import os

# Gunicorn Config
port = os.getenv("PORT", "8000")
bind = f"0.0.0.0:{port}"

# Workers Strategy
# 1 worker is ideal for async FastAPI/Uvicorn to minimize memory usage on Railway
# Can be customized via WEB_CONCURRENCY env variable
workers = int(os.getenv("WEB_CONCURRENCY", "1"))
worker_class = "uvicorn.workers.UvicornWorker"

# Timeouts (Important for long analysis)
timeout = 120
keepalive = 5

# Logging
loglevel = "info"
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr