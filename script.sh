nohup python main.py --port 8000 > main.log 2>&1 &
nohup python application.py --port 8001 > application.log 2>&1 &
nohup python app-cpu2.py --port 8002 > app_cpu2.log 2>&1 &
