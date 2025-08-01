import json
import os

import cv2
import easyocr
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------------
# Environment Setup
# -------------------------------
load_dotenv()
BASE_URL = os.getenv("BASE_URL")
LLM_KEY = os.getenv("LLM_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")

if not BASE_URL or not LLM_KEY:
    raise ValueError("Missing OpenAI BASE_URL or LLM_KEY in environment variables.")

# -------------------------------
# Initialize Tesseract & OpenAI
# -------------------------------
openai_client = OpenAI(base_url=BASE_URL, api_key=LLM_KEY)

# ---------------------------
# OCR Text Extraction (VietOCR)
# ---------------------------
def extract_text(image) -> str:
    """
    Converts an input OpenCV image to PIL format and extracts text using VietOCR.
    """
    # Preprocess: Convert BGR (OpenCV) to RGB then to PIL Image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    reader = easyocr.Reader(['vi', 'en'], gpu=False)  # Supports Vietnamese and English
    results = reader.readtext(gray, detail=0)
    
    # Join all text fragments into one string
    text = "\n".join(results)
    return text.strip()


# ---------------------------
# LLM-based Intelligent Parser
# ---------------------------
def extract_structured_info_with_llm(text: str) -> dict:
    """
    Uses an LLM to extract structured ID card fields from raw OCR text.
    Returns a JSON object with standardized fields, and auto-corrects the full name using PhoBERT.
    """
    prompt = f"""
        You are an expert at reading Vietnamese ID cards.
        Given the OCR-extracted raw text below, extract the following fields in JSON format:

        - Full Name (should be a plausible Vietnamese name)
        - Date of Birth (format: YYYY-MM-DD if possible)
        - ID Number
        - Address
        - Nationality
        - Gender

        If a field is missing, use null.
        Please keep content in vietnamese

        Raw OCR text:
        {text}
    """.strip()

    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        reply = response.choices[0].message.content.strip()
        result = json.loads(reply)
        return result

    except Exception as e:
        print(f"[LLM] Extraction error: {e}")
        return {"error": str(e)}
