Start-Process -FilePath "python" -ArgumentList "C:\files\git\AiHomeProxy\app.py" -NoNewWindow
Start-Sleep -Seconds 3
Invoke-WebRequest -Uri "http://127.0.0.1:5010/health" -UseBasicParsing
