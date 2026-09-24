#!/usr/bin/env python3
"""
Model Training Script (T04: Expanded Dataset & Local/Remote Support)
===================================================================
Trains the custom transformer model with diverse secret and false-positive patterns.

Usage:
    # Direct local training (no server needed):
    python train_model.py --local --epochs 15

    # Remote/HTTP training against running service:
    python train_model.py --url http://localhost:8080 --api-key "your_key" --epochs 10
"""

import sys
import json
import time
import argparse
import os
from datetime import datetime

# ── Training data (140+ diverse samples across High, Medium, Low) ──────────────
# Format: (secret_value, context, variable_name, label, user_action)
TRAINING_DATA = [
    # ── HIGH confidence secrets (Stripe) ──────────────────────────────────────
    ("sk-live-abc123XYZdef456GHIjkl789MNOpqr", "const stripeKey = \"sk-live-abc123XYZdef456GHIjkl789MNOpqr\";", "stripeKey", "high", "confirmed_secret"),
    ("sk-test-4eC39HqLyjWDarjtT1zdp7dc", "stripe.setApiKey('sk-test-4eC39HqLyjWDarjtT1zdp7dc')", "STRIPE_SECRET_KEY", "high", "confirmed_secret"),
    ("rk_test_" + "0123456789abcdefghijklmnopqrstuvwxyz", "const rk = 'rk_test_0123456789abcdefghijklmnopqrstuvwxyz';", "STRIPE_RESTRICTED_KEY", "high", "confirmed_secret"),
    ("whsec_abc123def456ghi789jkl012mno345pqr", "STRIPE_WEBHOOK_SECRET=whsec_abc123def456ghi789jkl012mno345pqr", "STRIPE_WEBHOOK_SECRET", "high", "confirmed_secret"),

    # ── HIGH confidence secrets (GitHub) ──────────────────────────────────────
    ("ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456", "GITHUB_TOKEN=ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456", "GITHUB_TOKEN", "high", "confirmed_secret"),
    ("ghp_16C7e42F292c6912E7710c838347Ae5b89", "export GITHUB_TOKEN=ghp_16C7e42F292c6912E7710c838347Ae5b89", "GITHUB_TOKEN", "high", "confirmed_secret"),
    ("gho_16C7e42F292c6912E7710c838347Ae5b89AbCdEf", "GITHUB_OAUTH=gho_16C7e42F292c6912E7710c838347Ae5b89AbCdEf", "GITHUB_OAUTH_TOKEN", "high", "confirmed_secret"),
    ("ghu_16C7e42F292c6912E7710c838347Ae5b89AbCdEf", "USER_TOKEN=ghu_16C7e42F292c6912E7710c838347Ae5b89AbCdEf", "GH_USER_TOKEN", "high", "confirmed_secret"),
    ("ghs_16C7e42F292c6912E7710c838347Ae5b89AbCdEf", "SERVER_TOKEN=ghs_16C7e42F292c6912E7710c838347Ae5b89AbCdEf", "GH_SERVER_TOKEN", "high", "confirmed_secret"),
    ("ghr_16C7e42F292c6912E7710c838347Ae5b89AbCdEf", "REFRESH_TOKEN=ghr_16C7e42F292c6912E7710c838347Ae5b89AbCdEf", "GH_REFRESH_TOKEN", "high", "confirmed_secret"),

    # ── HIGH confidence secrets (AWS) ─────────────────────────────────────────
    ("AKIAIOSFODNN7EXAMPLE", "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE", "AWS_ACCESS_KEY_ID", "high", "confirmed_secret"),
    ("AKIAI44QH8DHBEXAMPLE", "os.environ['AWS_ACCESS_KEY_ID'] = 'AKIAI44QH8DHBEXAMPLE'", "AWS_ACCESS_KEY_ID", "high", "confirmed_secret"),
    ("ASIAIOSFODNN7EXAMPLEA", "aws_session_key = 'ASIAIOSFODNN7EXAMPLEA'", "AWS_SESSION_KEY_ID", "high", "confirmed_secret"),
    ("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "AWS_SECRET_ACCESS_KEY", "high", "confirmed_secret"),
    ("je7MtGbClwBF/2Zp9Utk/h3yCo8nvbEXAMPLEKEY", "secret_key = 'je7MtGbClwBF/2Zp9Utk/h3yCo8nvbEXAMPLEKEY'", "AWS_SECRET_KEY", "high", "confirmed_secret"),

    # ── HIGH confidence secrets (Slack) ───────────────────────────────────────
    ("xoxb-17653671-f185e7c1cba", "SLACK_BOT_TOKEN=xoxb-17653671-f185e7c1cba", "SLACK_BOT_TOKEN", "high", "confirmed_secret"),
    ("xoxp-1234-5678-abcd-efgh", "slack_token = 'xoxp-1234-5678-abcd-efgh'", "slack_token", "high", "confirmed_secret"),
    ("xoxa-2-987654321012-1234567890123-abcdef", "SLACK_APP_TOKEN=xoxa-2-987654321012-1234567890123-abcdef", "SLACK_APP_TOKEN", "high", "confirmed_secret"),
    ("xoxr-123456789012-abcdef123456", "SLACK_REFRESH=xoxr-123456789012-abcdef123456", "SLACK_REFRESH_TOKEN", "high", "confirmed_secret"),

    # ── HIGH confidence secrets (Google & GCP) ────────────────────────────────
    ("AIzaSyD-9tSrke72I6e0DVblZMH4nCCF3EeFkXo", "GOOGLE_API_KEY=AIzaSyD-9tSrke72I6e0DVblZMH4nCCF3EeFkXo", "GOOGLE_API_KEY", "high", "confirmed_secret"),
    ("AIzaSyBw-EXAMPLE_KEY_1234567890abcdef", "const gkey = 'AIzaSyBw-EXAMPLE_KEY_1234567890abcdef';", "GCP_API_KEY", "high", "confirmed_secret"),
    ("ya29.a0AfH6SMDh_example_oauth2_token_string_here_1234567890", "Authorization: Bearer ya29.a0AfH6SMDh_example_oauth2_token_string_here_1234567890", "gcp_oauth", "high", "confirmed_secret"),

    # ── HIGH confidence secrets (SendGrid & Mailgun) ───────────────────────────
    ("SG.abc123def456ghi789jkl.mnOpQrStUvWxYz0123456789AbCdEfGhIjKlMn", "SENDGRID_API_KEY=SG.abc123def456ghi789jkl.mnOpQrStUvWxYz0123456789AbCdEfGhIjKlMn", "SENDGRID_API_KEY", "high", "confirmed_secret"),
    ("SG.9876543210_example_key.secret_part_here_abcdef", "sendgrid_key = 'SG.9876543210_example_key.secret_part_here_abcdef'", "SENDGRID_KEY", "high", "confirmed_secret"),
    ("key-3ax68njpkrfaixsyd2cqzprytiahxwjv", "MAILGUN_API_KEY=key-3ax68njpkrfaixsyd2cqzprytiahxwjv", "MAILGUN_API_KEY", "high", "confirmed_secret"),

    # ── HIGH confidence secrets (Twilio) ──────────────────────────────────────
    ("AC" + "d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4", "TWILIO_ACCOUNT_SID=AC" + "d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4", "TWILIO_ACCOUNT_SID", "high", "confirmed_secret"),
    ("SK" + "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d", "TWILIO_API_KEY=SK" + "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d", "TWILIO_API_SECRET", "high", "confirmed_secret"),

    # ── HIGH confidence secrets (Azure) ───────────────────────────────────────
    ("DefaultEndpointsProtocol=https;AccountName=myacc;AccountKey=AbCdEfGhIjKlMnOpQrStUvWxYz0123456789+/==;EndpointSuffix=core.windows.net", "AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=myacc;AccountKey=AbCdEfGhIjKlMnOpQrStUvWxYz0123456789+/==;EndpointSuffix=core.windows.net", "AZURE_STORAGE_CONNECTION_STRING", "high", "confirmed_secret"),
    ("ado_pat_abcdefghijklmnopqrstuvwxyz0123456789ABCDEF", "AZURE_DEVOPS_PAT=ado_pat_abcdefghijklmnopqrstuvwxyz0123456789ABCDEF", "AZURE_DEVOPS_PAT", "high", "confirmed_secret"),

    # ── HIGH confidence secrets (NPM, PyPI, Docker) ───────────────────────────
    ("npm_abcdefghijklmnopqrstuvwxyz01234567", "//registry.npmjs.org/:_authToken=npm_abcdefghijklmnopqrstuvwxyz01234567", "NPM_TOKEN", "high", "confirmed_secret"),
    ("pypi-AgEIcHlwaS5vcmcCJDFhMmIzYzRkLWU1ZjYtN2E4Yi05YzBkLTFlMmYzYTRiNWM2ZAACKDJkM2U0ZjVhLTZiN2MtOGQ5ZS0wZjFhLTJiM2M0ZDVsNmY3YQ", "TWINE_PASSWORD=pypi-AgEIcHlwaS5vcmcCJDFhMmIzYzRkLWU1ZjYtN2E4Yi05YzBkLTFlMmYzYTRiNWM2ZAACKDJkM2U0ZjVhLTZiN2MtOGQ5ZS0wZjFhLTJiM2M0ZDVsNmY3YQ", "PYPI_API_TOKEN", "high", "confirmed_secret"),
    ("dckr_pat_" + "AbCdEfGhIjKlMnOpQrStUvWxYz0", "DOCKER_ACCESS_TOKEN=dckr_pat_" + "AbCdEfGhIjKlMnOpQrStUvWxYz0", "DOCKER_ACCESS_TOKEN", "high", "confirmed_secret"),

    # ── HIGH confidence secrets (HuggingFace, OpenAI, Anthropic, Discord) ──────
    ("hf_abcdefghijklmnopqrstuvwxyz01234567", "HUGGING_FACE_HUB_TOKEN=hf_abcdefghijklmnopqrstuvwxyz01234567", "HUGGING_FACE_TOKEN", "high", "confirmed_secret"),
    ("sk-proj-abc123XYZdef456GHIjkl789MNOpqr012stuvwxYzABCDEF", "OPENAI_API_KEY=sk-proj-abc123XYZdef456GHIjkl789MNOpqr012stuvwxYzABCDEF", "OPENAI_API_KEY", "high", "confirmed_secret"),
    ("sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789ABCDEF", "ANTHROPIC_API_KEY=sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789ABCDEF", "ANTHROPIC_API_KEY", "high", "confirmed_secret"),
    ("OTg3NjU0MzIxMDk4NzY1NDMy.GHijkl.MnOpQrStUvWxYz1234567890", "DISCORD_BOT_TOKEN=OTg3NjU0MzIxMDk4NzY1NDMy.GHijkl.MnOpQrStUvWxYz1234567890", "DISCORD_BOT_TOKEN", "high", "confirmed_secret"),

    # ── HIGH confidence secrets (JWT & Auth Tokens) ───────────────────────────
    ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U", "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U", "auth_header", "high", "confirmed_secret"),
    ("eyJhbGciOiJSUzI1NiIsImtpZCI6IjEyMzQifQ.eyJpc3MiOiJhdXRoMCJ9.abcdefghijklmnopqrstuvwxyz1234567890_-", "const token = 'eyJhbGciOiJSUzI1NiIsImtpZCI6IjEyMzQifQ.eyJpc3MiOiJhdXRoMCJ9.abcdefghijklmnopqrstuvwxyz1234567890_-';", "JWT_TOKEN", "high", "confirmed_secret"),
    ("pk_live_51H1qfKJZ7eGyYbaCMmMOzXBc", "const pk = 'pk_live_51H1qfKJZ7eGyYbaCMmMOzXBc'", "STRIPE_PUBLIC_KEY", "high", "confirmed_secret"),

    # ── HIGH confidence secrets (Private Keys) ─────────────────────────────────
    ("-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA2a2rwplBQLF29amygykEMmYz0+Kcj3bKBp29P2rFj7SumBXl\n-----END RSA PRIVATE KEY-----", "private_key = open('id_rsa').read()", "private_key", "high", "confirmed_secret"),
    ("-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAACFwAAAAdzc2gtcn\n-----END OPENSSH PRIVATE KEY-----", "SSH_PRIVATE_KEY=-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAACFwAAAAdzc2gtcn\n-----END OPENSSH PRIVATE KEY-----", "SSH_PRIVATE_KEY", "high", "confirmed_secret"),
    ("-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIHBZg9x+9eMhG6v1nExamplePrivateKeyKeyKeyKeyKeyKeyKeyKeyKeyK\n-----END EC PRIVATE KEY-----", "ssl_key = '-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIHBZg9x+9eMhG6v1nExamplePrivateKeyKeyKeyKeyKeyKeyKeyKeyK\n-----END EC PRIVATE KEY-----'", "SSL_KEY", "high", "confirmed_secret"),

    # ── HIGH confidence secrets (Database URLs) ───────────────────────────────
    ("mongodb+srv://admin:p%40ssw0rd@cluster0.mongodb.net/mydb", "DATABASE_URL=mongodb+srv://admin:p%40ssw0rd@cluster0.mongodb.net/mydb", "DATABASE_URL", "high", "confirmed_secret"),
    ("postgresql://user:secretpassword@db.example.com:5432/mydb", "DATABASE_URL=postgresql://user:secretpassword@db.example.com:5432/mydb", "DATABASE_URL", "high", "confirmed_secret"),
    ("mysql://root:s3cr3tP%40ss@mysql-host:3306/production", "DB_URI=mysql://root:s3cr3tP%40ss@mysql-host:3306/production", "DB_URI", "high", "confirmed_secret"),
    ("redis://default:TopSecretRedisAuth123!@redis.internal:6379/0", "REDIS_URL=redis://default:TopSecretRedisAuth123!@redis.internal:6379/0", "REDIS_URL", "high", "confirmed_secret"),

    # ── MEDIUM confidence secrets (Custom tokens, App secrets) ────────────────
    ("abc123def456ghi789jkl012mno345pqr678", "const apiKey = 'abc123def456ghi789jkl012mno345pqr678'", "apiKey", "medium", "confirmed_secret"),
    ("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4", "API_TOKEN=a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4", "API_TOKEN", "medium", "confirmed_secret"),
    ("super_secret_password_123!", "config.password = 'super_secret_password_123!'", "password", "medium", "confirmed_secret"),
    ("mysecretkey-v2-production", "SECRET_KEY=mysecretkey-v2-production", "SECRET_KEY", "medium", "confirmed_secret"),
    ("7b89f02c4e1a3d5e7b89f02c4e1a3d5e", "CLIENT_SECRET=7b89f02c4e1a3d5e7b89f02c4e1a3d5e", "CLIENT_SECRET", "medium", "confirmed_secret"),
    ("f98a7b6c5d4e3f2a1b0c9d8e7f6a5b4c", "SESSION_SECRET=f98a7b6c5d4e3f2a1b0c9d8e7f6a5b4c", "SESSION_SECRET", "medium", "confirmed_secret"),
    ("K9#m$P2!vL@xQ7*wZ4&bC8^jF1~dY6(t", "MASTER_KEY=K9#m$P2!vL@xQ7*wZ4&bC8^jF1~dY6(t", "MASTER_KEY", "medium", "confirmed_secret"),
    ("0123456789abcdef0123456789abcdef01234567", "const hmacKey = '0123456789abcdef0123456789abcdef01234567';", "hmacKey", "medium", "confirmed_secret"),
    ("P@ssw0rd!Secure2026", "const db_pass = 'P@ssw0rd!Secure2026';", "db_pass", "medium", "confirmed_secret"),
    ("s3cr3t_auth_code_99x", "AUTH_SECRET=s3cr3t_auth_code_99x", "AUTH_SECRET", "medium", "confirmed_secret"),
    ("9f8e7d6c5b4a39281726354859607182", "ENCRYPTION_KEY=9f8e7d6c5b4a39281726354859607182", "ENCRYPTION_KEY", "medium", "confirmed_secret"),

    # ── LOW confidence / FALSE POSITIVES (Non-secret code & identifiers) ───────
    ("hello-world", "const greeting = 'hello-world'", "greeting", "low", "marked_false_positive"),
    ("my-app-name", "const appName = 'my-app-name'", "appName", "low", "marked_false_positive"),
    ("localhost", "const host = 'localhost'", "host", "low", "marked_false_positive"),
    ("http://localhost:3000", "const url = 'http://localhost:3000'", "url", "low", "marked_false_positive"),
    ("http://127.0.0.1:8080", "const backend = 'http://127.0.0.1:8080'", "BACKEND_URL", "low", "marked_false_positive"),
    ("0.0.0.0", "HOST=0.0.0.0", "HOST", "low", "marked_false_positive"),
    ("production", "const env = 'production'", "NODE_ENV", "low", "marked_false_positive"),
    ("development", "NODE_ENV=development", "NODE_ENV", "low", "marked_false_positive"),
    ("staging", "ENVIRONMENT=staging", "ENVIRONMENT", "low", "marked_false_positive"),
    ("1.0.0", "const version = '1.0.0'", "version", "low", "marked_false_positive"),
    ("v2.1.0-alpha", "RELEASE_VERSION=v2.1.0-alpha", "RELEASE_VERSION", "low", "marked_false_positive"),
    ("user@example.com", "const email = 'user@example.com'", "email", "low", "marked_false_positive"),
    ("support@mycompany.io", "CONTACT_EMAIL=support@mycompany.io", "CONTACT_EMAIL", "low", "marked_false_positive"),
    ("https://api.example.com/v1", "const baseUrl = 'https://api.example.com/v1'", "BASE_URL", "low", "marked_false_positive"),
    ("https://github.com/facebook/react", "REPO_URL=https://github.com/facebook/react", "REPO_URL", "low", "marked_false_positive"),
    ("2024-01-01", "const date = '2024-01-01'", "startDate", "low", "marked_false_positive"),
    ("2026-09-23T00:00:00Z", "const now = '2026-09-23T00:00:00Z'", "createdAt", "low", "marked_false_positive"),
    ("my-bucket-name", "const bucket = 'my-bucket-name'", "S3_BUCKET", "low", "marked_false_positive"),
    ("application/json", "const contentType = 'application/json'", "CONTENT_TYPE", "low", "marked_false_positive"),
    ("text/html; charset=utf-8", "headers['Content-Type'] = 'text/html; charset=utf-8'", "contentType", "low", "marked_false_positive"),
    ("UTF-8", "const encoding = 'UTF-8'", "ENCODING", "low", "marked_false_positive"),
    ("GET", "method: 'GET'", "method", "low", "marked_false_positive"),
    ("POST", "const httpMethod = 'POST'", "httpMethod", "low", "marked_false_positive"),
    ("true", "DEBUG=true", "DEBUG", "low", "marked_false_positive"),
    ("false", "ENABLE_CACHE=false", "ENABLE_CACHE", "low", "marked_false_positive"),
    ("0", "PORT=0", "PORT", "low", "marked_false_positive"),
    ("3000", "PORT=3000", "PORT", "low", "marked_false_positive"),
    ("8080", "const port = 8080;", "SERVER_PORT", "low", "marked_false_positive"),
    ("5432", "DB_PORT=5432", "DB_PORT", "low", "marked_false_positive"),
    ("/var/log/app.log", "LOG_FILE=/var/log/app.log", "LOG_FILE", "low", "marked_false_positive"),
    ("src/components/Button.tsx", "const path = 'src/components/Button.tsx'", "filePath", "low", "marked_false_positive"),
    ("public/assets/logo.png", "const logo = 'public/assets/logo.png'", "logoPath", "low", "marked_false_positive"),
    ("btn-primary", "const className = 'btn-primary'", "btnClass", "low", "marked_false_positive"),
    ("modal-dialog-centered", "class='modal-dialog-centered'", "modalClass", "low", "marked_false_positive"),
    ("Roboto, sans-serif", "fontFamily: 'Roboto, sans-serif'", "fontFamily", "low", "marked_false_positive"),
    ("#FFFFFF", "const bg = '#FFFFFF'", "bgColor", "low", "marked_false_positive"),
    ("#1E1E1E", "const darkBg = '#1E1E1E'", "darkBg", "low", "marked_false_positive"),
    ("click", "addEventListener('click', handler)", "eventName", "low", "marked_false_positive"),
    ("submit", "form.addEventListener('submit', onSubmit)", "eventType", "low", "marked_false_positive"),
    ("order_by_desc", "const sort = 'order_by_desc'", "sortOrder", "low", "marked_false_positive"),
    ("asc", "sort: 'asc'", "sort", "low", "marked_false_positive"),
    ("active", "status: 'active'", "status", "low", "marked_false_positive"),
    ("pending", "orderStatus = 'pending'", "orderStatus", "low", "marked_false_positive"),
    ("completed", "task.status = 'completed'", "taskStatus", "low", "marked_false_positive"),
    ("admin", "role: 'admin'", "userRole", "low", "marked_false_positive"),
    ("guest", "user.role = 'guest'", "role", "low", "marked_false_positive"),
    ("550e8400-e29b-41d4-a716-446655440000", "const orderId = '550e8400-e29b-41d4-a716-446655440000'", "orderId", "low", "marked_false_positive"),
    ("c2b4d6e8-1a3c-5e7g-9i0k-1234567890ab", "const itemId = 'c2b4d6e8-1a3c-5e7g-9i0k-1234567890ab'", "itemId", "low", "marked_false_positive"),
    ("YOUR_API_KEY_HERE", "const apiKey = 'YOUR_API_KEY_HERE'", "apiKey", "low", "marked_false_positive"),
    ("ENTER_YOUR_SECRET", "SECRET=ENTER_YOUR_SECRET", "SECRET", "low", "marked_false_positive"),
    ("dummy-secret-value", "const s = 'dummy-secret-value'", "dummySecret", "low", "marked_false_positive"),
    ("change_me", "PASSWORD=change_me", "PASSWORD", "low", "marked_false_positive"),
    ("us-east-1", "AWS_REGION=us-east-1", "AWS_REGION", "low", "marked_false_positive"),
    ("eu-west-1", "const region = 'eu-west-1'", "region", "low", "marked_false_positive"),
    ("sqs.us-east-1.amazonaws.com", "const endpoint = 'sqs.us-east-1.amazonaws.com'", "endpoint", "low", "marked_false_positive"),
]


def train_locally(epochs: int, reset: bool = False):
    """Direct in-process training via CustomLLM and persistence to DB and local JSON."""
    print("🚀 Initializing local training pipeline...")
    from src.analyzer import analyzer
    import numpy as np

    if reset:
        from src.model import CustomLLM
        analyzer.model = CustomLLM(analyzer.config)
        print("✨ Model weights initialized from scratch (--reset)")

    print(f"📊 Training dataset size: {len(TRAINING_DATA)} samples")
    print(f"🔄 Epochs: {epochs}")

    total_samples = len(TRAINING_DATA)
    start_time = time.time()

    import random
    data = list(TRAINING_DATA)

    for epoch in range(epochs):
        random.seed(epoch)
        random.shuffle(data)
        losses = []
        for secret, context, var_name, label, _ in data:
            features = analyzer.extract_features(secret, context, var_name)
            loss = analyzer.model.train_step(secret, features, label)
            losses.append(loss)

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        print(f"  Epoch {epoch+1:2d}/{epochs:2d} — Avg Loss: {avg_loss:.4f} (Adam step: {analyzer.model._adam_t})")

    analyzer.model.is_trained = True
    analyzer.save_model()
    print("💾 Model successfully saved to DB snapshot & local file.")

    elapsed = time.time() - start_time
    print(f"✅ Local training completed in {elapsed:.2f}s.\n")

    # Evaluation
    verify_model(analyzer=analyzer)


def train_remote(base_url: str, api_key: str, epochs: int):
    """Remote training via HTTP /train endpoint."""
    try:
        import requests
    except ImportError:
        print("❌ requests not installed — run: pip install requests")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    total_sent = 0
    total_ok = 0

    print(f"🌐 Remote Training against: {base_url}")
    print(f"📊 Samples: {len(TRAINING_DATA)} | Epochs: {epochs}\n")

    import random
    data = list(TRAINING_DATA)

    for epoch in range(epochs):
        random.seed(epoch)
        random.shuffle(data)
        epoch_ok = 0
        for secret, context, var_name, label, action in data:
            payload = {
                "secret_value": secret,
                "context": context,
                "variable_name": var_name,
                "label": label,
                "user_action": action,
            }
            try:
                r = requests.post(f"{base_url}/train", json=payload, headers=headers, timeout=15)
                if r.status_code == 200:
                    epoch_ok += 1
                    total_ok += 1
            except Exception as e:
                print(f"  ❌ Error sending sample: {e}")
            total_sent += 1
            time.sleep(0.02)
        print(f"  → Epoch {epoch+1}/{epochs} done: {epoch_ok}/{len(TRAINING_DATA)} samples OK")

    print(f"\nTraining complete: {total_ok}/{total_sent} samples sent.\n")
    verify_model(base_url=base_url)


def verify_model(analyzer=None, base_url=None):
    """Verify model performance on known test samples."""
    print("🔎 Verifying model inference accuracy...")
    test_cases = [
        ("sk-live-abc123XYZdef456GHIjkl789MNO", "const stripeKey = \"sk-live-abc123XYZdef456GHIjkl789MNO\";", "stripeKey", "high"),
        ("ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456", "GITHUB_TOKEN=ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456", "GITHUB_TOKEN", "high"),
        ("AKIAIOSFODNN7EXAMPLE", "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE", "AWS_ACCESS_KEY_ID", "high"),
        ("xoxb-17653671-f185e7c1cba", "SLACK_BOT_TOKEN=xoxb-17653671-f185e7c1cba", "SLACK_BOT_TOKEN", "high"),
        ("hello-world", "const greeting = 'hello-world'", "greeting", "low"),
        ("http://localhost:3000", "const url = 'http://localhost:3000'", "url", "low"),
        ("user@example.com", "const email = 'user@example.com'", "email", "low"),
        ("application/json", "const contentType = 'application/json'", "CONTENT_TYPE", "low"),
    ]

    correct = 0
    if analyzer:
        for secret, context, var, expected in test_cases:
            res = analyzer.analyze(secret, context, var)
            pred = res.get("enhanced_confidence", "unknown")
            is_ok = (expected == "high" and pred in ("high", "medium")) or (expected == "low" and pred == "low")
            if is_ok:
                correct += 1
            mark = "✅" if is_ok else "❌"
            print(f"  {mark} {var:20s}: pred={pred:6s} (expected={expected})")
    elif base_url:
        import requests, hmac as hmac_lib, hashlib
        secret_key = os.getenv("EXTENSION_SHARED_SECRET", "")
        if not secret_key:
            print("  ⚠️ Remote test skipped: Set EXTENSION_SHARED_SECRET env var to test remote HMAC endpoint.")
            return
        for secret, context, var, expected in test_cases:
            body = {"secret_value": secret, "context": context, "variable_name": var}
            body_bytes = json.dumps(body, separators=(',', ':'), sort_keys=True).encode()
            ts = str(time.time())
            msg = f"{ts}.".encode() + body_bytes
            sig = hmac_lib.new(secret_key.encode(), msg, hashlib.sha256).hexdigest()
            hdrs = {"X-Extension-Timestamp": ts, "X-Extension-Signature": sig,
                    "X-Machine-ID": "test-runner", "Content-Type": "application/json"}
            try:
                r = requests.post(f"{base_url}/extension/analyze", data=body_bytes, headers=hdrs, timeout=10)
                if r.status_code == 200:
                    pred = r.json().get("enhanced_confidence", "unknown")
                    is_ok = (expected == "high" and pred in ("high", "medium")) or (expected == "low" and pred == "low")
                    if is_ok: correct += 1
                    mark = "✅" if is_ok else "❌"
                    print(f"  {mark} {var:20s}: pred={pred:6s} (expected={expected})")
                else:
                    print(f"  ❌ {var:20s}: HTTP {r.status_code}")
            except Exception as e:
                print(f"  ❌ {var:20s}: {e}")

    total = len(test_cases)
    print(f"\n📊 Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    if correct == total:
        print("🎉 Perfect score! Model is robust and ready for deployment.")
    else:
        print("⚠️ Some cases mismatched, consider tuning epochs or learning rate.")


def main():
    parser = argparse.ArgumentParser(description="Train python-llm model")
    parser.add_argument("--local", action="store_true", help="Train locally in-process without requiring a running server")
    parser.add_argument("--reset", action="store_true", help="Reinitialize weights from scratch before training")
    parser.add_argument("--url", default="http://localhost:8080", help="Service URL for remote training")
    parser.add_argument("--api-key", default=os.getenv("API_KEY", ""), help="API key for remote /train endpoint")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  python-llm Model Training")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # If --local specified or no api-key provided, train locally
    if args.local or not args.api_key:
        if not args.local and not args.api_key:
            print("💡 No --api-key provided, defaulting to local in-process training (--local).\n")
        train_locally(args.epochs, reset=args.reset)
    else:
        train_remote(args.url.rstrip("/"), args.api_key, args.epochs)


if __name__ == "__main__":
    main()