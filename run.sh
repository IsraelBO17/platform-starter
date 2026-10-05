#!/bin/bash
# run.sh - the Lambda Web Adapter starts the function by running this script.
exec python -m uvicorn app.main:app --host 0.0.0.0 --port "${AWS_LWA_PORT:-8080}"
