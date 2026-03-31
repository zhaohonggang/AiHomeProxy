import requests
import json

url = "http://127.0.0.1:5010/v1/chat/completions"
data = {
    "model": "Home-0.0.1",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
}

try:
    response = requests.post(url, json=data, timeout=30)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
except requests.exceptions.ConnectionError:
    print("Error: Cannot connect to server. Is it running?")
except Exception as e:
    print(f"Error: {e}")
