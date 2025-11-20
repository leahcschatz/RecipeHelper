from flask import Flask, render_template, request, jsonify
import openai
import io
import json
import os
import re
import unicodedata
import logging
from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import easyocr
import numpy as np
import cv2
import ssl
import certifi
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    from google.cloud import vision
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False

# Fix SSL certificate issues for EasyOCR model downloads (macOS)
try:
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl._create_default_https_context = lambda: ssl_context
except Exception:
    pass  # If certifi not available, use system defaults

from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import trafilatura


# --- App setup ---
app = Flask(__name__)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Google Cloud Vision setup
GOOGLE_VISION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if GOOGLE_VISION_AVAILABLE and GOOGLE_VISION_CREDENTIALS:
    try:
        # Initialize Google Vision client
        vision_client = vision.ImageAnnotatorClient()
        app.logger.info("Google Cloud Vision API initialized successfully")
    except Exception as e:
        app.logger.warning(f"Google Cloud Vision initialization failed: {e}")
        vision_client = None
else:
    vision_client = None
    if GOOGLE_VISION_AVAILABLE:
        app.logger.info("Google Cloud Vision available but credentials not configured")

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
        
        CRITICAL: Only extract information that is ACTUALLY present in the text. Do NOT invent, guess, or hallucinate any ingredients, amounts, or instructions. If the text is unclear or incomplete, extract only what you can clearly see. If you cannot find a recipe in the text, return an empty recipe with title "Unable to extract recipe".
        
        IMPORTANT RULES FOR INGREDIENTS:
        1. Preserve ALL section headers within ingredients (e.g., "For the sauce:", "For the marinade:", "For the dough:", etc.). 
           Include these headers as separate entries in the ingredients list.
        2. If metric and imperial measurements are given, use both.
        3. Keep ingredient quantities and units exactly as written (e.g., ½ cup, 1 tbsp, 200 g).
        4. Only include ingredients that are explicitly mentioned in the text.
        
        IMPORTANT RULES FOR INSTRUCTIONS:
        1. Whenever an instruction mentions an ingredient, append the ingredient amount in parentheses next to it.
        2. If an instruction says something like "mix all ingredients for [component]" or "combine all [component] ingredients" 
           or "add all ingredients from the [component] section", EXPAND it to list ALL the ingredients for that component 
           with their amounts. For example, if an instruction says "mix all ingredients for the sauce", replace it with 
           "mix tomatoes (1 cup), olive oil (2 tbsp), and garlic (2 cloves) for the sauce".
        3. If no specific amount is mentioned for an ingredient in an instruction, assume it's the full amount from the ingredient section.
        4. Only include instructions that are explicitly present in the text.
        
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


def is_valid_word(word):
    """Check if a word looks like a real word (not OCR garbage)."""
    if len(word) < 2:
        return False
    # Word should have mostly letters
    alpha_count = sum(1 for c in word if c.isalpha())
    if alpha_count < len(word) * 0.5:  # At least 50% letters
        return False
    # Word shouldn't be mostly numbers or symbols
    if sum(1 for c in word if c.isdigit()) > len(word) * 0.5:
        return False
    return True


def filter_garbage_text(text):
    """Filter out obvious OCR garbage and keep only readable text."""
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip lines that are mostly numbers/symbols
        alpha_chars = sum(1 for c in line if c.isalpha())
        if len(line) > 0 and alpha_chars / len(line) < 0.3:
            continue
        
        # Skip very short lines that are just symbols
        if len(line) < 3:
            continue
        
        # Skip lines that are mostly single characters separated by spaces (OCR artifact)
        words = line.split()
        if len(words) > 5 and all(len(w) == 1 for w in words[:5]):
            continue
        
        # Keep lines that have some real words
        valid_words = [w for w in words if is_valid_word(w)]
        if len(valid_words) > 0:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def preprocess_phone_image(img_array, method='light'):
    """Enhanced preprocessing for phone camera photos."""
    # Convert to OpenCV format (BGR)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    if method == 'light':
        # Light preprocessing - just denoise and enhance contrast slightly
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Light denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)
        # Convert back to RGB
        processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        return processed
    elif method == 'moderate':
        # Moderate preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return processed
    else:
        # Aggressive preprocessing (original)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        adaptive_thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        processed = cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2RGB)
        return processed


def try_paddleocr(img_array):
    """Try PaddleOCR (often best for phone photos)."""
    if not hasattr(extract_text_from_file, '_paddleocr_reader'):
        app.logger.info("Initializing PaddleOCR reader (first time only)...")
        extract_text_from_file._paddleocr_reader = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
    
    reader = extract_text_from_file._paddleocr_reader
    result = reader.ocr(img_array, cls=True)
    
    # Extract text from results
    text_lines = []
    if result and result[0]:
        for line in result[0]:
            if line and len(line) > 1:
                text_lines.append(line[1][0])  # Get text from [bbox, (text, confidence)]
    
    return '\n'.join(text_lines)


def try_easyocr(img_array):
    """Try EasyOCR."""
    if not hasattr(extract_text_from_file, '_easyocr_reader'):
        app.logger.info("Initializing EasyOCR reader (first time only)...")
        extract_text_from_file._easyocr_reader = easyocr.Reader(['en'], gpu=False)
    
    reader = extract_text_from_file._easyocr_reader
    results = reader.readtext(img_array)
    return '\n'.join([result[1] for result in results])


def try_pytesseract(img_array):
    """Try pytesseract with preprocessing."""
    # Convert numpy array back to PIL Image
    image = Image.fromarray(img_array)
    
    # Additional preprocessing for pytesseract
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Use better config for recipes
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?()[]{}/-+*=&%$#@ '
    return pytesseract.image_to_string(image, config=custom_config)


def try_google_vision(file_bytes):
    """Try Google Cloud Vision API (best for phone photos)."""
    if not vision_client:
        raise Exception("Google Vision client not initialized")
    
    # Create image object from bytes
    image = vision.Image(content=file_bytes)
    
    # Perform text detection
    response = vision_client.text_detection(image=image)
    texts = response.text_annotations
    
    if texts:
        # The first annotation contains the entire detected text
        full_text = texts[0].description
        return full_text
    else:
        return ""


def resize_image_if_needed(image, max_dimension=2000):
    """Resize image if too large to save memory. Maintains aspect ratio."""
    width, height = image.size
    if width <= max_dimension and height <= max_dimension:
        return image
    
    # Calculate new size maintaining aspect ratio
    if width > height:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    else:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))
    
    app.logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height} to save memory")
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def extract_text_from_file(file_bytes, mime_type, filename):
    """Extract text from a file (PDF or image) using OCR."""
    if "pdf" in mime_type:
        app.logger.info(f"Processing PDF file: {filename}")
        # Reduce DPI for memory efficiency
        pages = convert_from_bytes(file_bytes, dpi=150)
        text = ""
        custom_config = r'--oem 3 --psm 6'
        for page in pages:
            # Resize if too large
            page = resize_image_if_needed(page, max_dimension=2000)
            # Preprocess PDF pages similar to images
            page = page.convert('L')  # Grayscale
            enhancer = ImageEnhance.Contrast(page)
            page = enhancer.enhance(2.0)  # Increase contrast
            page = page.filter(ImageFilter.SHARPEN)
            page = page.convert('RGB')
            text += pytesseract.image_to_string(page, config=custom_config) + "\n"
        app.logger.info(f"Extracted {len(text)} characters from PDF {filename}")
        return text
    elif "image" in mime_type:
        app.logger.info(f"Processing image file: {filename}")
        try:
            # Load and resize image to save memory
            image = Image.open(io.BytesIO(file_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large (saves significant memory)
            image = resize_image_if_needed(image, max_dimension=2000)
            
            img_array = np.array(image)
            app.logger.info(f"Image loaded: {image.size[0]}x{image.size[1]} pixels, mode: {image.mode}")
            
            # Try Google Cloud Vision first (no local models, most memory efficient)
            if vision_client:
                try:
                    app.logger.info("Trying Google Vision (memory efficient, no local models)")
                    text = try_google_vision(file_bytes)
                    if text and len(text.strip()) > 10:
                        app.logger.info(f"✓ Google Vision SUCCESS: {len(text)} characters extracted")
                        return text
                except Exception as e:
                    app.logger.warning(f"Google Vision failed: {e}, trying local OCR...")
            
            # If Google Vision not available or failed, try local OCR methods one at a time
            # Only create one processed version at a time to save memory
            text = None
            best_text = None
            best_score = 0
            
            # Try EasyOCR first (often best for phone photos, but loads models)
            try:
                app.logger.info("Trying EasyOCR (original image)")
                text = try_easyocr(img_array)
                if text:
                    words = text.split()
                    word_count = len([w for w in words if is_valid_word(w)])
                    app.logger.info(f"EasyOCR extracted {len(text)} chars, {word_count} valid words")
                    if word_count > best_score:
                        best_text = text
                        best_score = word_count
                    if word_count >= 10:  # Good enough, stop here
                        app.logger.info(f"✓ EasyOCR SUCCESS: {len(text)} characters extracted")
                        return text
            except Exception as e:
                app.logger.warning(f"EasyOCR failed: {e}")
            
            # Try pytesseract (lightweight, no model loading)
            try:
                app.logger.info("Trying Pytesseract (lightweight)")
                light_processed = preprocess_phone_image(img_array, method='light')
                text = try_pytesseract(light_processed)
                if text:
                    words = text.split()
                    word_count = len([w for w in words if is_valid_word(w)])
                    app.logger.info(f"Pytesseract extracted {len(text)} chars, {word_count} valid words")
                    if word_count > best_score:
                        best_text = text
                        best_score = word_count
                    if word_count >= 10:
                        app.logger.info(f"✓ Pytesseract SUCCESS: {len(text)} characters extracted")
                        # Clear memory before returning
                        del light_processed
                        del img_array
                        return text
                # Clear memory
                del light_processed
            except Exception as e:
                app.logger.warning(f"Pytesseract failed: {e}")
            
            # Use best result if we have one
            text = best_text
            # Clear memory
            del img_array
            
            if not text or len(text.strip()) < 10:
                app.logger.error(f"All OCR methods failed. Text length: {len(text) if text else 0}")
                raise Exception("All OCR methods failed to extract sufficient text")
            
            app.logger.info(f"✓ Best result: {len(text)} chars, {best_score} valid words")
            
            # Filter out garbage and validate text quality
            original_length = len(text)
            text = filter_garbage_text(text)
            app.logger.info(f"After filtering: {original_length} -> {len(text)} chars")
            
            words = text.split()
            valid_words = [w for w in words if is_valid_word(w)]
            total_chars = len(text)
            alpha_chars = sum(1 for c in text if c.isalpha())
            alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0
            
            app.logger.info(f"Text quality metrics: {len(valid_words)} valid words, {alpha_ratio:.2%} alphabetic chars")
            app.logger.info(f"Filtered text preview: {text[:500]}...")
            
            if len(valid_words) < 10 or alpha_ratio < 0.3:
                app.logger.error(f"Extracted text appears to be garbage after filtering.")
                app.logger.error(f"Valid words: {len(valid_words)}, Alpha ratio: {alpha_ratio:.2%}")
                raise Exception(f"OCR extracted invalid text. Image may be too blurry, low quality, or unreadable. Please try:\n1. Taking a clearer photo with better lighting\n2. Ensuring the text is in focus\n3. Using a higher resolution image")
            
            # Log detailed text information
            text_length = len(text)
            text_lines = len(text.split('\n'))
            app.logger.info(f"OCR SUCCESS: Extracted {text_length} characters in {text_lines} lines")
            
            return text
        except Exception as e:
            app.logger.error(f"Error processing image {filename}: {e}")
            raise Exception(f"Failed to process image: {str(e)}")
    else:
        app.logger.warning(f"Unsupported file type: {mime_type} for {filename}")
        return None


def call_openai_for_recipe(prompt, extracted_text):
    """Call OpenAI API to extract recipe data from text."""
    try:
        app.logger.info(f"Sending data to OpenAI... (text length: {len(extracted_text)} chars)")
        
        # Log a sample of what we're sending
        sample_text = extracted_text[:300] if len(extracted_text) > 300 else extracted_text
        app.logger.info(f"Sample text being sent to OpenAI: {sample_text[:200]}...")
        
        full_content = prompt + "\n\n" + extracted_text
        app.logger.info(f"Total content length: {len(full_content)} characters")
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": full_content}]
        )
        
        response_text = response.choices[0].message.content
        app.logger.info(f"OpenAI response received: {len(response_text)} characters")
        app.logger.info(f"OpenAI response preview: {response_text[:500]}...")
        
        return response_text
    except Exception as e:
        app.logger.exception("OpenAI API error.")
        raise Exception(f"OpenAI API error: {str(e)}")


def parse_recipe_response(response_text):
    """Parse OpenAI response into recipe data dictionary."""
    app.logger.info(f"Parsing OpenAI response...")
    try:
        recipe_data = json.loads(response_text)
        app.logger.info(f"Successfully parsed JSON. Title: {recipe_data.get('title', 'N/A')}")
        app.logger.info(f"Ingredients count: {len(recipe_data.get('ingredients', []))}")
        app.logger.info(f"Instructions count: {len(recipe_data.get('instructions', []))}")
    except json.JSONDecodeError as e:
        app.logger.warning(f"JSON parse error: {e}. Attempting regex extraction...")
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            try:
                recipe_data = json.loads(json_match.group(0))
                app.logger.info("Successfully extracted JSON using regex")
            except json.JSONDecodeError as e2:
                app.logger.error(f"Regex extraction also failed: {e2}")
                recipe_data = {
                    "title": "Unknown",
                    "ingredients": [],
                    "instructions": []
                }
        else:
            app.logger.error("No JSON found in response")
            recipe_data = {
                "title": "Unknown",
                "ingredients": [],
                "instructions": []
            }
    
    # Log if recipe appears empty or has error message
    title = recipe_data.get('title', '')
    if 'unable to extract' in title.lower() or not title or title == 'Unknown':
        app.logger.warning(f"Recipe extraction may have failed. Title: '{title}'")
        app.logger.warning(f"Full recipe data: {json.dumps(recipe_data, indent=2)}")
    
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
    app.logger.info(f"Processing recipe text (is_web_page={is_web_page})...")
    
    # Normalize text
    original_length = len(extracted_text)
    extracted_text = unicodedata.normalize("NFKC", extracted_text)
    app.logger.info(f"Text normalized: {original_length} -> {len(extracted_text)} chars")
    
    # Get appropriate prompt
    prompt = get_recipe_parsing_prompt() if is_web_page else get_recipe_extraction_prompt()
    app.logger.info(f"Using {'web page' if is_web_page else 'file extraction'} prompt")
    
    # Call OpenAI
    response_text = call_openai_for_recipe(prompt, extracted_text)
    
    # Parse response
    recipe_data = parse_recipe_response(response_text)
    
    # Clean recipe data
    recipe_data = clean_recipe_data(recipe_data)
    
    # Final validation logging
    title = recipe_data.get('title', '')
    ingredients_count = len(recipe_data.get('ingredients', []))
    instructions_count = len(recipe_data.get('instructions', []))
    
    app.logger.info(f"Final recipe data - Title: '{title}', Ingredients: {ingredients_count}, Instructions: {instructions_count}")
    
    if not title or title == 'Unknown' or ingredients_count == 0 or instructions_count == 0:
        app.logger.warning("WARNING: Recipe appears incomplete or empty!")
        app.logger.warning(f"Full recipe structure: {json.dumps(recipe_data, indent=2)}")
    
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