#!/bin/bash

echo "Starting Social Media Addiction Prediction API..."
echo "API will be available at: http://localhost:8000"
echo "API documentation at: http://localhost:8000/docs"
echo ""

uvicorn app:app --host 0.0.0.0 --port 8000 --reload
