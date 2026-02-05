import uvicorn
import logging
import requests
import io
import time
import os
from fastapi import FastAPI, Request
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions, AnalyzeImageOptions, ImageData
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from pypdf import PdfReader
from docx import Document
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")

# Initialize FastAPI
app = FastAPI()

# Initialize Azure Client
if not AZURE_ENDPOINT or not AZURE_KEY:
    raise ValueError("Please set AZURE_ENDPOINT and AZURE_KEY")

client = ContentSafetyClient(AZURE_ENDPOINT, AzureKeyCredential(AZURE_KEY))

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_severity(categories_analysis, block_level=2):
    """
    Helper function to check if any category exceeds the block level.
    """
    for item in categories_analysis:
        if item.severity >= block_level:
            return True, f"{item.category} (Level {item.severity})"
    return False, ""

@app.post("/webhook")
async def handle_webhook(request: Request):
    """
    Handles CometChat requests for Text, Images, and Files.
    """
    start_time = time.time()
    try:
        payload = await request.json()
        context_list = payload.get('contextMessages', [])
        
        if not context_list:
            return {"isMatchingCondition": False}

        # 1. Get Current Message Data
        last_entry = context_list[-1]
        sender_uid = next(iter(last_entry))
        message_object = last_entry[sender_uid]
        
        # 🚨 CRITICAL FIX: Check if it is a Dict or String to prevent Attribute Error
        msg_type = 'text' # Default to text
        if isinstance(message_object, dict):
            msg_type = message_object.get('type', 'text')
        
        violation_found = False
        reason_msg = ""
        
        # ==========================================
        # 📷 PATH A: IMAGE MODERATION
        # ==========================================
        if msg_type == 'image' and isinstance(message_object, dict):
            data = message_object.get('data', {})
            image_url = data.get('url')
            
            if image_url:
                logger.info(f"📷 Analyzing Image: {image_url}")
                response = requests.get(image_url)
                if response.status_code == 200:
                    try:
                        # Use Raw Bytes
                        image_data = ImageData(content=response.content)
                        request_options = AnalyzeImageOptions(image=image_data)
                        
                        # Analyze
                        result = client.analyze_image(request_options)
                        
                        # Check Severity (Level 2 for Images)
                        violation_found, reason_msg = analyze_severity(result.categories_analysis, block_level=2)
                        
                    except HttpResponseError as e:
                        logger.error(f"Azure Image API Failed: {e.error.message if e.error else e}")
                else:
                    logger.warning("Could not download image")

        # ==========================================
        # 📂 PATH B: FILE MODERATION (PDF / DOCX)
        # ==========================================
        elif msg_type == 'file' and isinstance(message_object, dict):
            data = message_object.get('data', {})
            file_url = data.get('url')
            file_ext = data.get('extension', '').lower()
            
            if file_url:
                logger.info(f"📂 Analyzing File ({file_ext})")

                response = requests.get(file_url)
                if response.status_code == 200:
                    extracted_text = ""
                    file_memory = io.BytesIO(response.content)
                    
                    # Extract Text
                    if 'pdf' in file_ext:
                        try:
                            reader = PdfReader(file_memory)
                            # LIMIT: Only read first 3 pages for speed
                            for page in reader.pages[:3]: 
                                extracted_text += page.extract_text() + " "
                        except Exception as e:
                            logger.error(f"PDF Error: {e}")
                    elif 'doc' in file_ext:
                        try:
                            doc = Document(file_memory)
                            for para in doc.paragraphs:
                                extracted_text += para.text + " "
                        except Exception as e:
                            logger.error(f"Doc Error: {e}")
                    elif 'txt' in file_ext:
                        extracted_text = response.text

                    # Send to Azure (if text found)
                    if extracted_text.strip():
                        try:
                            # Truncate to 1000 chars for speed
                            request_options = AnalyzeTextOptions(text=extracted_text[:1000])
                            result = client.analyze_text(request_options)
                            violation_found, reason_msg = analyze_severity(result.categories_analysis, block_level=2)
                        except HttpResponseError as e:
                            logger.error(f"Azure Text API Failed: {e}")

        # ==========================================
        # 💬 PATH C: TEXT MODERATION (Fallback)
        # ==========================================
        else:
            combined_text = ""
            # Loop safely through mixed Strings and Dicts
            for entry in context_list:
                for key, value in entry.items():
                    if isinstance(value, str):
                        combined_text += f"{value}. "
                    elif isinstance(value, dict) and 'data' in value:
                        current_text = value['data'].get('text', '')
                        combined_text += f"{current_text}. "
            
            # Only analyze if text exists
            if combined_text.strip():
                logger.info(f"🧐 Analyzing Text: {combined_text[:50]}...")
                try:
                    request_options = AnalyzeTextOptions(text=combined_text[:1000])
                    result = client.analyze_text(request_options)
                    violation_found, reason_msg = analyze_severity(result.categories_analysis, block_level=4) # Level 4 for text
                except HttpResponseError as e:
                     logger.error(f"Azure Text API Failed: {e}")

        # ==========================================
        # 🏁 RETURN DECISION
        # ==========================================
        duration = time.time() - start_time
        logger.info(f"⏱️ Time: {duration:.2f}s")

        if violation_found:
            logger.warning(f"🚫 BLOCKED: {reason_msg}")
            return {
                "isMatchingCondition": True,
                "confidence": 0.95,
                "reason": reason_msg
            }
        
        logger.info("✅ Safe")
        return {
            "isMatchingCondition": False,
            "confidence": 0.98,
            "reason": "Safe"
        }

    except Exception as e:
        logger.error(f"❌ Critical Error: {e}")
        return {"isMatchingCondition": False}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)