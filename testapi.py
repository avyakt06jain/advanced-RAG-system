import requests
import json
import time

# --- Configuration ---
# The URL where your FastAPI application is running
API_URL = "http://127.0.0.1:8000/hackrx/run"

# The Bearer token for authorization
API_KEY = "06864514c746f45fb93a6e0421a052c7875d3d1fd841d870f397c9d50e4146f8"

# The headers for the request
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# The payload (body) of the request
payload = {
    "document": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "queries": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

def test_rag_api():
    """
    Sends a POST request to the RAG API endpoint and prints the response.
    """
    print(f"Sending request to: {API_URL}")
    print("-" * 30)
    
    start_time = time.time()
    
    try:
        # Send the POST request
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=300) # 5 minute timeout
        
        end_time = time.time()
        print(f"Request completed in {end_time - start_time:.2f} seconds.")
        print("-" * 30)

        # Check the response status code
        if response.status_code == 200:
            print("✅ Request successful (Status Code: 200)")
            print("\n--- API Response ---")
            # Pretty-print the JSON response
            response_data = response.json()
            print(json.dumps(response_data, indent=2))
        else:
            print(f"❌ Request failed with Status Code: {response.status_code}")
            print("\n--- Error Response ---")
            try:
                # Try to print the JSON error detail if it exists
                error_data = response.json()
                print(json.dumps(error_data, indent=2))
            except json.JSONDecodeError:
                # If the response is not JSON, print the raw text
                print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the request: {e}")

if __name__ == "__main__":
    test_rag_api()