"""
Gemini-based visual analysis for query processing
"""
import google.generativeai as genai
import os
import cv2
import numpy as np
from typing import Dict, List, Any
import json

class GeminiAnalyzer:
    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def analyze_people_in_frame(self, frame: np.ndarray, people_boxes: List, query: str) -> List[bool]:
        """
        Analyze each person and determine if they match the query
        Returns list of booleans indicating which people to anonymize
        """
        results = []
        
        for idx, (x, y, w, h) in enumerate(people_boxes):
            person_crop = frame[y:y+h, x:x+w]
            
            # Convert to RGB and encode
            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', person_rgb)
            
            # Create prompt
            prompt = f"""Analyze this person and answer whether they match this criteria: "{query}"

Return ONLY a JSON object with this exact structure:
{{
    "matches": true or false,
    "reasoning": "brief explanation",
    "attributes": {{
        "gender": "male/female/unknown",
        "approximate_age_group": "child/teen/adult/elderly",
        "visible_emotions": "happy/sad/angry/neutral/etc",
        "clothing_colors": ["color1", "color2"]
    }}
}}

Be accurate and specific. Focus on visible attributes only."""

            try:
                # Generate response
                response = self.model.generate_content([
                    prompt,
                    {"mime_type": "image/jpeg", "data": buffer.tobytes()}
                ])
                
                # Parse JSON response
                text = response.text.strip()
                # Remove markdown code blocks if present
                if text.startswith('```json'):
                    text = text[7:]
                if text.startswith('```'):
                    text = text[3:]
                if text.endswith('```'):
                    text = text[:-3]
                
                data = json.loads(text.strip())
                results.append(data.get('matches', False))
                
            except Exception as e:
                print(f"Gemini analysis error for person {idx}: {e}")
                # Default to not anonymizing on error
                results.append(False)
        
        return results