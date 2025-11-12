import json

def query_gemini(model, json_data, prompt):
    context = json_data.get("context", "")
    persons = json_data.get("persons", [])

    query_text = f"""
    Scene Context: "{context}"
    People Data: {json.dumps(persons, indent=2)}
    Task: {"Decide which people should be blurred based on the context." if not prompt else f"Find people matching: {prompt}"}
    Return only their 'id' numbers in a valid JSON array.
    """

    response = model.generate_content(query_text)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```json")[-1].split("```")[-1].strip()

    try:
        result = json.loads(text)
        if isinstance(result, list) and all(isinstance(i, int) for i in result):
            print(result)
            return result
        print([])
        return []
    except json.JSONDecodeError:
        print([])
        return []
