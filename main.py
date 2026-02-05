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

def analyze_severity(categories_analysis, block_level=4):
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

        # Get Current Message Data
        current_msg_entry = context_list[-1]
        sender_uid = next(iter(current_msg_entry))
        message_object = current_msg_entry[sender_uid]
        msg_type = message_object.get('type', 'text')
        
        violation_found = False
        reason_msg = ""
        
        # ==========================================
        # 📷 PATH A: IMAGE MODERATION
        # ==========================================
        if msg_type == 'image':
            image_url = message_object['data']['url']
            logger.info(f"📷 Analyzing Image: {image_url}")
            
            # Download the image
            response = requests.get(image_url)
            if response.status_code == 200:
                try:
                    # Use Raw Bytes (Matches Azure Docs)
                    image_data = ImageData(content=response.content)
                    request_options = AnalyzeImageOptions(image=image_data)
                    
                    # Analyze
                    result = client.analyze_image(request_options)
                    
                    # Check Severity (Strict Level 4)
                    violation_found, reason_msg = analyze_severity(result.categories_analysis, block_level=4)
                    
                except HttpResponseError as e:
                    logger.error(f"Azure Image API Failed: {e.error.message if e.error else e}")
            else:
                logger.warning("Could not download image from CometChat")

        # ==========================================
        # 📂 PATH B: FILE MODERATION (PDF / DOCX)
        # ==========================================
        elif msg_type == 'file':
            file_url = message_object['data']['url']
            file_ext = message_object['data'].get('extension', '').lower()
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
                        violation_found, reason_msg = analyze_severity(result.categories_analysis, block_level=4)
                    except HttpResponseError as e:
                        logger.error(f"Azure Text API Failed: {e}")

        # ==========================================
        # 💬 PATH C: TEXT MODERATION (Original Logic)
        # ==========================================
        else:
            combined_text = ""
            for entry in context_list:
                for key, value in entry.items():
                    if isinstance(value, str):
                        combined_text += f"{value}. "
                    elif isinstance(value, dict) and 'data' in value:
                        current_text = value['data'].get('text', '')
                        combined_text += f"{current_text}"
            
            logger.info(f"🧐 Analyzing Text: {combined_text[:50]}...")
            
            try:
                request_options = AnalyzeTextOptions(text=combined_text[:1000])
                result = client.analyze_text(request_options)
                violation_found, reason_msg = analyze_severity(result.categories_analysis, block_level=4)
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
                "confidence": 100,
                "reason": reason_msg
            }
        
        logger.info("✅ Safe")
        return {
            "isMatchingCondition": False,
            "confidence": 0,
            "reason": "Safe"
        }

    except Exception as e:
        logger.error(f"❌ Critical Error: {e}")
        # Fail Open to prevent blocking good users on error
        return {"isMatchingCondition": False}

if __name__ == "__main__":
    # Use 10000 for Render, or 5000 for local testing
    uvicorn.run(app, host="0.0.0.0", port=10000)