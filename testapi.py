import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/hackrx/run")
API_KEY = os.getenv("API_KEY", "your_api_key_here")

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

payload = {
    "document": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "queries": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
    ],
}


def test_rag_api():
    import requests

    print(f"Sending request to: {API_URL}")
    print("-" * 30)

    start_time = time.time()

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=300)
        end_time = time.time()
        print(f"Request completed in {end_time - start_time:.2f} seconds.")
        print("-" * 30)

        if response.status_code == 200:
            print("✅ Request successful (Status Code: 200)")
            print("\n--- API Response ---")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Request failed with Status Code: {response.status_code}")
            print("\n--- Error Response ---")
            try:
                print(json.dumps(response.json(), indent=2))
            except json.JSONDecodeError:
                print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the request: {e}")


if __name__ == "__main__":
    test_rag_api()
