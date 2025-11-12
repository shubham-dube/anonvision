import os
import google.generativeai as genai

def setup_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")
