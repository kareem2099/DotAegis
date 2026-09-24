#!/usr/bin/env python3
"""
LLM Secret Detection Service - Main Entry Point
===============================================

Production-ready FastAPI application for AI-powered secret detection.
"""

import os
import uvicorn
from src.service import app

if __name__ == "__main__":
    # Set development defaults if not set
    if not os.getenv('ENVIRONMENT'):
        os.environ['ENVIRONMENT'] = 'development'

    if not os.getenv('API_KEY'):
        # In production, set this via environment variable (e.g., in Railway or Docker)
        os.environ['API_KEY'] = ''

    if not os.getenv('JWT_SECRET'):
        # In production, set this via environment variable
        os.environ['JWT_SECRET'] = 'change-this-in-production-for-security'

    if not os.getenv('RATE_LIMIT_REQUESTS_PER_MINUTE'):
        os.environ['RATE_LIMIT_REQUESTS_PER_MINUTE'] = '60'

    if not os.getenv('LOG_LEVEL'):
        os.environ['LOG_LEVEL'] = 'INFO'

    if not os.getenv('LOG_FORMAT'):
        os.environ['LOG_FORMAT'] = 'console'  # Use 'json' for production

    # Get port from environment (Railway sets PORT automatically)
    port = int(os.getenv('PORT', '8000'))

    # Run the server
    uvicorn.run(
        "src.service:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv('ENVIRONMENT') == 'development',
        log_level=os.getenv('LOG_LEVEL', 'info').lower(),
        access_log=True
    )
