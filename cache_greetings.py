import requests
import json

greetings = [
    "Hey",
    "Hello",
    "Yo",
    "Hi there",
    "Greetings",
    "What's up",
    "How's it going"
]

url = "http://127.0.0.1:8200/generate"

for greeting in greetings:
    print(f"Caching: {greeting}")
    try:
        response = requests.post(url, json={"text": greeting, "language": "en"})
        if response.status_code == 200:
            print("  OK")
        else:
            print(f"  Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"  Error: {e}")