#!/usr/bin/env python3
"""
Generate secure API keys and secrets for Railway deployment.
Run this script to generate production-ready credentials.
"""

import secrets
import string
import json
import os
from datetime import datetime

def generate_secure_key(length=32):
    """Generate a cryptographically secure random key."""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def generate_api_key():
    """Generate a production-ready API key."""
    # API keys typically have a prefix for identification
    prefix = "llm_"
    key = generate_secure_key(28)  # Total length will be 32 with prefix
    return f"{prefix}{key}"

def generate_jwt_secret():
    """Generate a JWT secret (longer for better security)."""
    return generate_secure_key(64)

def main():
    """Generate and display production credentials."""
    print("🔐 Generating Secure Credentials for Railway Deployment")
    print("=" * 60)

    # Generate credentials
    api_key = generate_api_key()
    jwt_secret = generate_jwt_secret()

    # Display credentials
    print("\n📋 Copy these to your Railway Environment Variables:")
    print(f"API_KEY={api_key}")
    print(f"JWT_SECRET={jwt_secret}")
    print("ENVIRONMENT=production")
    print("LOG_LEVEL=INFO")
    print("RATE_LIMIT_REQUESTS_PER_MINUTE=100")
    # Save to a local file for reference (not for production use)
    credentials = {
        "generated_at": datetime.utcnow().isoformat(),
        "api_key": api_key,
        "jwt_secret": jwt_secret,
        "note": "These credentials were generated for Railway deployment. Keep them secure!"
    }

    with open('.railway-credentials.json', 'w') as f:
        json.dump(credentials, f, indent=2)

    print("\n💾 Credentials saved to .railway-credentials.json (for your reference only)")
    print("⚠️  IMPORTANT: Delete this file after copying credentials to Railway!")
    print("\n🚀 Next Steps:")
    print("1. Go to Railway Dashboard → Your Project → Variables")
    print("2. Add the environment variables shown above")
    print("3. Deploy your service!")
    print("\n🎉 Your LLM service will be production-ready in minutes!")
if __name__ == "__main__":
    main()
