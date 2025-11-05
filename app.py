from flask import Flask, render_template, request, jsonify
import openai, io, json, os, re, unicodedata, logging
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
from dotenv import load_dotenv
import logging


# --- App setup ---
app = Flask(__name__)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app.logger.addHandler(logging.StreamHandler())
app.logger.setLevel(logging.INFO)
app.logger.propagate = True

# --- Prompt ---
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

# --- Health route (for Render health checks) ---
@app.route("/health")
def health():
    return "OK", 200


# --- Index route ---
@app.route('/')
def index():
    return render_template("index.html")


# --- File processing route ---
@app.route('/process_file', methods=['POST'])
def process_file():
    print("📥 /process_file endpoint hit", flush=True)
    try:
        uploaded_file = request.files.get('file')
        if not uploaded_file:
            print("⚠️ No file uploaded", flush=True)
            app.logger.error("No file uploaded.")
            return jsonify({"error": "No file uploaded"}), 400

        mime_type = uploaded_file.mimetype.lower()
        file_bytes = uploaded_file.read()
        extracted_text = ""

        print(f"📄 Got file: {uploaded_file.filename} ({uploaded_file.mimetype})", flush=True)

        # --- Extract text ---
        if "pdf" in mime_type:
            app.logger.info("Processing PDF file...")
            pages = convert_from_bytes(file_bytes, dpi=100)
            for page in pages:
                extracted_text += pytesseract.image_to_string(page) + "\n"
        elif "image" in mime_type:
            app.logger.info("Processing image file...")
            image = Image.open(io.BytesIO(file_bytes))
            extracted_text = pytesseract.image_to_string(image)
        else:
            app.logger.warning(f"Unsupported file type: {mime_type}")
            return jsonify({"error": f"Unsupported file type: {mime_type}"}), 400

        if not extracted_text.strip():
            app.logger.error("Empty extracted text.")
            return jsonify({"error": "Could not extract text from file."}), 400

        extracted_text = unicodedata.normalize("NFKC", extracted_text)

        # --- OpenAI call ---
        try:
            app.logger.info("Sending data to OpenAI...")
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": PROMPT + "\n\n" + extracted_text}]
            )
            response_text = response.choices[0].message.content
        except Exception as e:
            app.logger.exception("OpenAI API error.")
            return jsonify({"error": f"OpenAI API error: {str(e)}"}), 500

        # --- Parse response ---
        try:
            recipe_data = json.loads(response_text)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            recipe_data = json.loads(json_match.group(0)) if json_match else {
                "title": "Unknown",
                "ingredients": [],
                "instructions": []
            }

        # --- Clean up steps ---
        cleaned_steps = []
        for step in recipe_data.get("instructions", []):
            cleaned = re.sub(r"^\s*\d+\.\s*", "", step)
            cleaned_steps.append(cleaned)

        recipe_data["instructions"] = cleaned_steps
        app.logger.info("Recipe processed successfully.")

        return render_template("recipe.html", recipe=recipe_data)

    except Exception as e:
        print(f"❌ Exception in /process_file: {e}", flush=True)
        app.logger.exception("Unexpected server error.")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# --- Main ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
