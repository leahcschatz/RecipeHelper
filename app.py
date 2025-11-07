from flask import Flask, render_template, request, jsonify
import openai
import io
import json
import os
import re
import unicodedata
import logging
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import trafilatura


# --- App setup ---
app = Flask(__name__)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app.logger.addHandler(logging.StreamHandler())
app.logger.setLevel(logging.INFO)
app.logger.propagate = True


# --- Helper Functions ---

def get_recipe_extraction_prompt():
    """Returns the prompt for recipe extraction from text."""
    return """
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


def get_recipe_parsing_prompt():
    """Returns the prompt for parsing recipes from web pages."""
    return """
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


def extract_text_from_file(file_bytes, mime_type, filename):
    """Extract text from a file (PDF or image) using OCR."""
    if "pdf" in mime_type:
        app.logger.info(f"Processing PDF file: {filename}")
        pages = convert_from_bytes(file_bytes, dpi=75)
        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page) + "\n"
        return text
    elif "image" in mime_type:
        app.logger.info(f"Processing image file: {filename}")
        image = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(image)
    else:
        app.logger.warning(f"Unsupported file type: {mime_type} for {filename}")
        return None


def call_openai_for_recipe(prompt, extracted_text):
    """Call OpenAI API to extract recipe data from text."""
    try:
        app.logger.info("Sending data to OpenAI...")
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt + "\n\n" + extracted_text}]
        )
        return response.choices[0].message.content
    except Exception as e:
        app.logger.exception("OpenAI API error.")
        raise Exception(f"OpenAI API error: {str(e)}")


def parse_recipe_response(response_text):
    """Parse OpenAI response into recipe data dictionary."""
    try:
        recipe_data = json.loads(response_text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        recipe_data = json.loads(json_match.group(0)) if json_match else {
            "title": "Unknown",
            "ingredients": [],
            "instructions": []
        }
    return recipe_data


def clean_recipe_data(recipe_data):
    """Clean and normalize recipe data."""
    # Clean up instruction steps (remove leading numbers)
    cleaned_steps = []
    for step in recipe_data.get("instructions", []):
        cleaned = re.sub(r"^\s*\d+\.\s*", "", step)
        cleaned_steps.append(cleaned)
    recipe_data["instructions"] = cleaned_steps
    return recipe_data


def process_recipe_text(extracted_text, is_web_page=False):
    """Process extracted text through OpenAI and return recipe data."""
    # Normalize text
    extracted_text = unicodedata.normalize("NFKC", extracted_text)
    
    # Get appropriate prompt
    prompt = get_recipe_parsing_prompt() if is_web_page else get_recipe_extraction_prompt()
    
    # Call OpenAI
    response_text = call_openai_for_recipe(prompt, extracted_text)
    
    # Parse response
    recipe_data = parse_recipe_response(response_text)
    
    # Clean recipe data
    recipe_data = clean_recipe_data(recipe_data)
    
    return recipe_data


# --- Routes ---

@app.route("/health")
def health():
    """Health check route for Render."""
    return "OK", 200


@app.route('/')
def index():
    """Index page route."""
    return render_template("index.html")


@app.route('/process_file', methods=['POST'])
def process_file():
    """Process uploaded file(s) and extract recipe."""
    print("📥 /process_file endpoint hit", flush=True)
    try:
        # Get all uploaded files (support multiple files)
        uploaded_files = request.files.getlist('file')
        if not uploaded_files or len(uploaded_files) == 0:
            print("⚠️ No files uploaded", flush=True)
            app.logger.error("No files uploaded.")
            return jsonify({"error": "No files uploaded"}), 400

        print(f"📄 Got {len(uploaded_files)} file(s)", flush=True)
        all_text_parts = []

        # Process each file
        for idx, uploaded_file in enumerate(uploaded_files):
            if not uploaded_file.filename:
                continue
                
            mime_type = uploaded_file.mimetype.lower()
            file_bytes = uploaded_file.read()

            print(f"📄 Processing file {idx + 1}/{len(uploaded_files)}: {uploaded_file.filename} ({uploaded_file.mimetype})", flush=True)

            # Extract text from file
            file_text = extract_text_from_file(file_bytes, mime_type, uploaded_file.filename)
            
            if file_text and file_text.strip():
                all_text_parts.append(file_text)
            else:
                app.logger.warning(f"Could not extract text from {uploaded_file.filename}")

        # Combine all extracted text
        if not all_text_parts:
            app.logger.error("No text extracted from any file.")
            return jsonify({"error": "Could not extract text from any file."}), 400

        # Join all text parts with page separators
        extracted_text = "\n\n--- Page Break ---\n\n".join(all_text_parts)
        app.logger.info(f"Combined text from {len(all_text_parts)} file(s), total length: {len(extracted_text)} characters")

        # Process recipe through OpenAI
        recipe_data = process_recipe_text(extracted_text, is_web_page=False)
        app.logger.info("Recipe processed successfully.")

        return render_template("recipe.html", recipe=recipe_data)

    except Exception as e:
        print(f"❌ Exception in /process_file: {e}", flush=True)
        app.logger.exception("Unexpected server error.")
        error_msg = str(e)
        if "OpenAI API error" in error_msg:
            return jsonify({"error": error_msg}), 500
        return jsonify({"error": f"Server error: {error_msg}"}), 500

def get_browser_headers():
    """Get standard browser headers for web requests."""
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/118.0 Safari/537.36"
        )
    }


def authenticate_nytimes(session):
    """Authenticate with NYTimes Cooking using credentials from .env"""
    username = os.getenv("NYTIMES_USERNAME")
    password = os.getenv("NYTIMES_PASSWORD")
    
    if not username or not password:
        raise ValueError("NYTimes credentials not found in .env file. Please set NYTIMES_USERNAME and NYTIMES_PASSWORD")
    
    headers = get_browser_headers()
    headers.update({
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://cooking.nytimes.com/",
    })
    
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

def extract_text_from_url(recipe_url, session):
    """Extract text from a recipe URL."""
    is_nytimes = "cooking.nytimes.com" in recipe_url.lower()
    
    # Fetch the page
    try:
        response = session.get(recipe_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch recipe: {e}")

    # Extract text based on source
    if is_nytimes:
        # Use trafilatura on authenticated HTML response
        extracted_text = trafilatura.extract(response.text)
        # Fallback to BeautifulSoup if trafilatura doesn't work
        if not extracted_text or len(extracted_text.strip()) == 0:
            soup = BeautifulSoup(response.text, 'html.parser')
            extracted_text = soup.get_text(separator='\n', strip=True)
    else:
        downloaded = trafilatura.fetch_url(recipe_url)
        extracted_text = trafilatura.extract(downloaded)

    if not extracted_text or len(extracted_text.strip()) == 0:
        raise Exception("Failed to extract readable text")
    
    return extracted_text


@app.route('/process_url', methods=['POST'])
def process_url():
    """Process recipe URL and extract recipe."""
    recipe_url = request.form.get('url')
    if not recipe_url:
        return "No URL provided", 400

    # Check if this is a NYTimes Cooking URL
    is_nytimes = "cooking.nytimes.com" in recipe_url.lower()
    
    # Set up session
    session = requests.Session()
    session.headers.update(get_browser_headers())
    
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
        # Extract text from URL
        extracted_text = extract_text_from_url(recipe_url, session)
        
        # Process recipe through OpenAI
        recipe_data = process_recipe_text(extracted_text, is_web_page=True)
        
        return render_template("recipe.html", recipe=recipe_data)
    except Exception as e:
        app.logger.error(f"Error processing URL: {e}")
        error_msg = str(e)
        if "OpenAI API error" in error_msg:
            return f"OpenAI API error: {error_msg}", 500
        return f"Error: {error_msg}", 400


# --- Main ---
if __name__ == "__main__":
    # port = int(os.environ.get("PORT", 5000))
    # app.run(host="0.0.0.0", port=port, debug=True)
    app.run(debug=True, port=5001, use_reloader=False)