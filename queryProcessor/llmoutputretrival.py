import json
import requests
import time
from dotenv import load_dotenv
import os

load_dotenv()


def create_llm_prompt(user_question, document_chunks):
    system_instruction = "You are a helpful assistant. Use the following document context to answer the user's question. If the answer is not in the context, state that you don't have enough information."
    context = "\n\n".join(document_chunks)
    full_prompt = f"{system_instruction}\n\nDocument Context:\n{context}\n\nUser Question: {user_question}\n\nAnswer:"
    return full_prompt


def generate_answer_from_llm(prompt):
    api_key = os.getenv("API_KEY")
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

    headers = {
        "Content-Type": "application/json",
    }

    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}

    try:
        print("Sending prompt to LLM...")
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # This will raise an exception for HTTP errors

        result = response.json()

        # Check if the response contains a valid candidate and text
        if result and "candidates" in result and result["candidates"]:
            # Extract and return the answer from the LLM response
            answer = result["candidates"][0]["content"]["parts"][0]["text"]
            return answer
        else:
            print("Received an unexpected response from the API.")
            return "Unable to generate an answer."

    except requests.exceptions.HTTPError as errh:
        print(f"Http Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Something went wrong: {err}")
    except KeyError:
        print("Received an invalid response format from the API.")
        # Print the full response for debugging
        print(json.dumps(result, indent=2))

    return "An error occurred while generating the answer."


# This block will run when the script is executed directly.
if __name__ == "__main__":
    original_prompt = "What are the health benefits of drinking green tea?"

    relevant_document_chunks = [
        "Green tea has been shown to improve various aspects of cardiovascular health. Studies indicate that regular consumption can help improve the function of blood vessels, leading to a reduced risk of heart disease.",
        "The tea also contains powerful antioxidants known as catechins. These compounds are effective at neutralizing free radicals and may play a role in preventing cell damage and the development of certain types of cancer.",
        "Furthermore, green tea can aid in weight management by boosting metabolism and fat oxidation. It also contains L-theanine, an amino acid that promotes relaxation without causing drowsiness.",
    ]

    # --- Step 2: Create the final LLM prompt ---
    llm_prompt = create_llm_prompt(original_prompt, relevant_document_chunks)

    print("--- Final Prompt for LLM ---")
    print(llm_prompt)
    print("-" * 30)

    # --- Step 3: Call the LLM to generate the answer ---
    final_answer = generate_answer_from_llm(llm_prompt)

    print("\n--- Final Generated Answer ---")
    print(final_answer)
    print("-" * 30)
