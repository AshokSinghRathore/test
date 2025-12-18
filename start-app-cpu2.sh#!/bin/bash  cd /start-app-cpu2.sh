#!/bin/bash

cd /workspace || exit 1

nohup uvicorn app-cpu2:app \
  --host 0.0.0.0 \
  --port 8002 \
  --workers 1 \
  > app_cpu2.log 2>&1 &
