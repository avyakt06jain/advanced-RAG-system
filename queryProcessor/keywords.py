import spacy

# Load the small English language model.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli.download import download

    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def extract_keywords(prompt):
    doc = nlp(prompt)
    keywords = []
    for token in doc:
        # Check if the token is not a stop word and is not punctuation.
        # We also check for alpha characters to filter out numbers and symbols.
        if not token.is_stop and not token.is_punct and token.is_alpha:
            if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
                keywords.append(token.lemma_)
            elif token.pos_ in ["VERB"]:
                keywords.append(token.lemma_)

    # important entities like people, locations, or product names.
    for ent in doc.ents:
        keywords.append(ent.text)

    # Convert the list to a set to remove duplicates, then back to a list.
    unique_keywords = list(set([kw for kw in keywords if len(kw) > 1]))

    return unique_keywords


# This block will only run when the script is executed directly.
if __name__ == "__main__":
    # Example prompts to test the function.
    test_prompts = [
        "What are the health benefits of drinking green tea?",
        "Please summarize the main points about the historical events of World War II.",
        "Tell me about the launch date of the new space telescope by NASA.",
        "Does this policy cover knee surgery, and what are the conditions?",
    ]

    for prompt in test_prompts:
        extracted = extract_keywords(prompt)
        print(f"Original Prompt: '{prompt}'")
        print(f"Extracted Keywords: {extracted}\n")
