#!/usr/bin/env python3
"""Rebuild the synthetic feature-schema-v2 release bootstrap.

Does not train the live service. Reviewed online updates use admin /train
or /feedback/review. See README for deployment and validation.
"""
from scripts.build_bootstrap import main

if __name__ == '__main__':
    main()
