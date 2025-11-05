from flask import Flask, render_template, request
import openai, io, json
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
from dotenv import load_dotenv
import os, openai
import unicodedata
import re

app = Flask(__name__)
load_dotenv()  # loads .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT = """
You are a recipe extraction assistant.
Extract the recipe title, ingredients and instructions from this text. For the ingredients if metric and imperial 
measurements are given use both and For the instructions section whenever it mentions an ingredient,
next to the ingredient put the amount that is needed in parentheses. If no amount is specific assume it's the full amount from the 
ingredient section. Return them in JSON format ONLY:

{
  "title": [string],
  "ingredients": [string],
  "instructions": [string]
}
"""

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/process_file', methods=['POST'])
def process_file():
    uploaded_file = request.files.get('file')
    if not uploaded_file:
        return "No file uploaded", 400

    mime_type = uploaded_file.mimetype.lower()
    file_bytes = uploaded_file.read()

    # Extract text from PDFs or images
    extracted_text = ""

    if "pdf" in mime_type:
        pages = convert_from_bytes(file_bytes)
        for page in pages:
            extracted_text += pytesseract.image_to_string(page) + "\n"
    elif "image" in mime_type:
        image = Image.open(io.BytesIO(file_bytes))
        extracted_text = pytesseract.image_to_string(image)
    else:
        return f"Unsupported file type: {mime_type}", 400

    if not extracted_text.strip():
        return "Could not extract text from file.", 400

    extracted_text = unicodedata.normalize("NFKC", extracted_text)

    # Send to OpenAI
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": PROMPT + "\n\n" + extracted_text}]
        )
        response_text = response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API error: {e}", 500

    # Parse JSON safely
    try:
        recipe_data = json.loads(response_text)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        recipe_data = json.loads(json_match.group(0)) if json_match else {
            "title": "Unknown",
            "ingredients": [],
            "instructions": []
        }

    cleaned_steps = []
    for step in recipe_data["instructions"]:
        cleaned = re.sub(r"^\s*\d+\.\s*", "", step)  # removes "1. " at start
        cleaned_steps.append(cleaned)

    recipe_data["instructions"] = cleaned_steps 

    return render_template("recipe.html", recipe=recipe_data)


if __name__ == "__main__":
     app.run(debug=True, use_reloader=False)
