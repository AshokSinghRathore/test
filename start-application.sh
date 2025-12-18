#!/bin/bash

cd /workspace || exit 1

nohup uvicorn application:app \
  --host 0.0.0.0 \
  --port 8001 \
  --workers 1 \
  > application.log 2>&1 &
