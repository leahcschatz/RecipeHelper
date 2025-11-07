from flask import Flask, render_template, request, jsonify
import openai, io, json, os, re, unicodedata, logging
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
from dotenv import load_dotenv
import logging
from fpdf import FPDF
import requests
from bs4 import BeautifulSoup


# --- App setup ---
app = Flask(__name__)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app.logger.addHandler(logging.StreamHandler())
app.logger.setLevel(logging.INFO)
app.logger.propagate = True


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
            pages = convert_from_bytes(file_bytes, dpi=75)
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

        # --- Prompt ---
        PROMPT = """
        You are a recipe extraction assistant.
        Extract the recipe title, ingredients and instructions from this text. 
        
        IMPORTANT RULES FOR INGREDIENTS:
        1. Preserve ALL section headers within ingredients (e.g., "For the sauce:", "For the marinade:", "For the dough:", etc.). 
           Include these headers as separate entries in the ingredients list.
        2. If metric and imperial measurements are given, use both.
        3. Keep ingredient quantities and units exactly as written (e.g., ½ cup, 1 tbsp, 200 g).
        
        IMPORTANT RULES FOR INSTRUCTIONS:
        1. Whenever an instruction mentions an ingredient, append the ingredient amount in parentheses next to it.
        2. If an instruction says something like "mix all ingredients for [component]" or "combine all [component] ingredients" 
           or "add all ingredients from the [component] section", EXPAND it to list ALL the ingredients for that component 
           with their amounts. For example, if an instruction says "mix all ingredients for the sauce", replace it with 
           "mix tomatoes (1 cup), olive oil (2 tbsp), and garlic (2 cloves) for the sauce".
        3. If no specific amount is mentioned for an ingredient in an instruction, assume it's the full amount from the ingredient section.
        
        Return them in JSON format ONLY:

        {
        "title": [string],
        "ingredients": [string],
        "instructions": [string]
        }
        """
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
    

import trafilatura

def authenticate_nytimes(session):
    """Authenticate with NYTimes Cooking using credentials from .env"""
    username = os.getenv("NYTIMES_USERNAME")
    password = os.getenv("NYTIMES_PASSWORD")
    
    if not username or not password:
        raise ValueError("NYTimes credentials not found in .env file. Please set NYTIMES_USERNAME and NYTIMES_PASSWORD")
    
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/118.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://cooking.nytimes.com/",
    }
    
    # First, get the login page to establish session and get any CSRF tokens
    login_page_url = "https://myaccount.nytimes.com/auth/login"
    try:
        page_response = session.get(login_page_url, headers=headers, timeout=10)
        app.logger.info(f"Login page response status: {page_response.status_code}")
    except Exception as e:
        app.logger.warning(f"Could not fetch login page: {e}")
    
    # Try different login endpoints - NYTimes may use different versions
    login_endpoints = [
        "https://myaccount.nytimes.com/svc/login/v2/login",
        "https://myaccount.nytimes.com/svc/login/v3/login",
        "https://myaccount.nytimes.com/svc/login/v1/login",
        "https://myaccount.nytimes.com/svc/login/login",
    ]
    
    login_data = {
        "login": username,
        "password": password
    }
    
    last_error = None
    for login_url in login_endpoints:
        try:
            headers_with_content = headers.copy()
            headers_with_content["Content-Type"] = "application/json"
            headers_with_content["Origin"] = "https://myaccount.nytimes.com"
            
            response = session.post(login_url, json=login_data, headers=headers_with_content, timeout=10)
            app.logger.info(f"Trying {login_url}, response status: {response.status_code}")
            
            if response.status_code == 200:
                # Check if login was successful
                try:
                    login_result = response.json()
                    if login_result.get("status") == "OK":
                        app.logger.info("NYTimes authentication successful")
                        return session
                    else:
                        error_msg = login_result.get("message", "Authentication failed")
                        last_error = f"NYTimes authentication failed: {error_msg}"
                        continue
                except (ValueError, KeyError):
                    # If response is not JSON, check if we got cookies which indicate successful login
                    if response.cookies:
                        app.logger.info("NYTimes authentication successful (cookies received)")
                        return session
                    else:
                        last_error = f"NYTimes authentication failed: No valid response from {login_url}"
                        continue
            elif response.status_code == 404:
                # Try next endpoint
                continue
            else:
                last_error = f"NYTimes authentication failed with status code: {response.status_code}"
                continue
                
        except requests.RequestException as e:
            last_error = f"NYTimes authentication request failed: {str(e)}"
            continue
    
    # If all endpoints failed, try form-based login as fallback
    try:
        app.logger.info("Trying form-based login as fallback...")
        form_login_url = "https://myaccount.nytimes.com/auth/login"
        form_data = {
            "username": username,
            "password": password,
            "login": username,
        }
        
        headers_form = headers.copy()
        headers_form["Content-Type"] = "application/x-www-form-urlencoded"
        headers_form["Origin"] = "https://myaccount.nytimes.com"
        
        response = session.post(form_login_url, data=form_data, headers=headers_form, timeout=10, allow_redirects=True)
        app.logger.info(f"Form login response status: {response.status_code}")
        
        # Check if we got redirected or got cookies
        if response.status_code in [200, 302] or response.cookies:
            app.logger.info("NYTimes authentication successful (form-based)")
            return session
    except Exception as e:
        app.logger.warning(f"Form-based login also failed: {e}")
    
    # If all methods failed, raise the last error
    if last_error:
        raise Exception(last_error)
    else:
        raise Exception("NYTimes authentication failed: All methods attempted failed")

@app.route('/process_url', methods=['POST'])
def process_url():
    recipe_url = request.form.get('url')
    if not recipe_url:
        return "No URL provided", 400

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/118.0 Safari/537.36"
        )
    }

    # Check if this is a NYTimes Cooking URL
    is_nytimes = "cooking.nytimes.com" in recipe_url.lower()
    
    session = requests.Session()
    session.headers.update(headers)
    
    # Authenticate if it's a NYTimes Cooking URL
    if is_nytimes:
        try:
            app.logger.info("Detected NYTimes Cooking URL, authenticating...")
            session = authenticate_nytimes(session)
            app.logger.info("NYTimes authentication successful")
        except Exception as e:
            app.logger.error(f"NYTimes authentication error: {e}")
            return f"Failed to authenticate with NYTimes: {e}", 400

    try:
        response = session.get(recipe_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return f"Failed to fetch recipe: {e}", 400

    # ✅ Cleanly extract main text
    # For NYTimes, use trafilatura on the authenticated HTML response
    if is_nytimes:
        # Use trafilatura to extract clean text from the authenticated HTML
        extracted_text = trafilatura.extract(response.text)
        # Fallback to BeautifulSoup if trafilatura doesn't work
        if not extracted_text or len(extracted_text.strip()) == 0:
            soup = BeautifulSoup(response.text, 'html.parser')
            extracted_text = soup.get_text(separator='\n', strip=True)
    else:
        downloaded = trafilatura.fetch_url(recipe_url)
        extracted_text = trafilatura.extract(downloaded)

    if not extracted_text or len(extracted_text.strip()) == 0:
        return "Failed to extract readable text", 400

    extracted_text = unicodedata.normalize("NFKC", extracted_text)

    # --- Send to OpenAI ---

    PROMPT = """
        You are a recipe parsing assistant.

        You will receive text extracted from a recipe web page. These pages often contain long stories,
        advertisements, notes, or unrelated sections. Ignore anything that is not part of the actual recipe.

        Your task is to:
        1. Identify the **recipe title**, **ingredients**, and **instructions** only.
        2. Ignore introductions, author notes, nutritional info, FAQs, comments, or links.
        3. Detect the start of the recipe when a section header like "Ingredients" appears.
        4. Stop parsing once the recipe ends (for example, before comments or social links).
        
        IMPORTANT RULES FOR INGREDIENTS:
        5. Preserve ALL section headers within ingredients (e.g., "For the sauce:", "For the marinade:", "For the dough:", 
           "For the pasta:", "For the filling:", etc.). Include these headers as separate entries in the ingredients list.
        6. Keep ingredient quantities and units exactly as written (e.g., ½ cup, 1 tbsp, 200 g).
        7. If metric and imperial measurements are given, use both.
        
        IMPORTANT RULES FOR INSTRUCTIONS:
        8. If an instruction mentions an ingredient, append the ingredient quantity in parentheses next to it.
        9. If an instruction says something like "mix all ingredients for [component]" or "combine all [component] ingredients" 
           or "add all ingredients from the [component] section" or "use all [component] ingredients", EXPAND it to list 
           ALL the ingredients for that component with their amounts. 
           For example, if an instruction says "mix all ingredients for the sauce", replace it with 
           "mix tomatoes (1 cup), olive oil (2 tbsp), and garlic (2 cloves) for the sauce".
        10. If no specific amount is mentioned for an ingredient in an instruction, assume it's the full amount from the ingredient section.

        Return only valid JSON in the following format:

        {
        "title": "Recipe Title",
        "ingredients": [
            "For the sauce:",
            "1 cup tomatoes",
            "2 tbsp olive oil",
            "2 cloves garlic",
            "For the pasta:",
            "1 lb pasta",
            "2 cups water"
        ],
        "instructions": [
            "Mix tomatoes (1 cup), olive oil (2 tbsp), and garlic (2 cloves) for the sauce.",
            "Cook pasta (1 lb) in water (2 cups) until al dente."
        ]
        }
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": PROMPT + "\n\n" + extracted_text}]
        )
        response_text = response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API error: {e}", 500

    # --- Parse JSON safely ---
    try:
        recipe_data = json.loads(response_text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        recipe_data = json.loads(json_match.group(0)) if json_match else {
            "title": "Unknown",
            "ingredients": [],
            "instructions": []
        }

    return render_template("recipe.html", recipe=recipe_data)


# --- Main ---
if __name__ == "__main__":
    # port = int(os.environ.get("PORT", 5000))
    # app.run(host="0.0.0.0", port=port, debug=True)
    app.run(debug=True, port=5001, use_reloader=False)