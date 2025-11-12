import json
from geminiAPI_handle import setup_gemini
from gemini_handler import query_gemini

def main():
    with open("people_data.json", "r") as f:
        json_data = json.load(f)
    
    prompt = input("Enter your query: ").strip()
    model = setup_gemini()

    result = query_gemini(model, json_data, prompt)

    print("\nFiltered People:\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
