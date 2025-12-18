#!/bin/bash

cd /workspace || exit 1

nohup uvicorn main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  > main.log 2>&1 &
