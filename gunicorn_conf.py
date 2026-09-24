import multiprocessing
import os

# Gunicorn Config
port = os.getenv("PORT", "8000")
bind = f"0.0.0.0:{port}"

# Workers Strategy
# (2 x Cores) + 1 is the standard formula for CPU-bound tasks
workers = 4
# workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"

# Timeouts (Important for long analysis)
timeout = 120
keepalive = 5

# Logging
loglevel = "info"
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr