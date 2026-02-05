import uvicorn
import logging
import requests
import time
import os
from fastapi import FastAPI, Request
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions, AnalyzeImageOptions, ImageData
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")

# Initialize FastAPI and Azure
app = FastAPI()
client = ContentSafetyClient(AZURE_ENDPOINT, AzureKeyCredential(AZURE_KEY))

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_safety(analysis_result, block_level):
    """Simple helper to check if severity is too high."""
    for item in analysis_result.categories_analysis:
        if item.severity >= block_level:
            return True, f"{item.category} (Level {item.severity})"
    return False, ""

@app.post("/webhook")
async def handle_webhook(request: Request):
    start_time = time.time()
    try:
        payload = await request.json()
        context_list = payload.get('contextMessages', [])
        
        if not context_list:
            return {"isMatchingCondition": False}

        # 1. Get the current message safely
        last_entry = context_list[-1]
        
        # The entry looks like: { "uid_123": { "type": "image", ... } }
        # We need to get that inner dictionary.
        current_msg_val = next(iter(last_entry.values()))

        # 2. Determine Type (The Critical Fix)
        msg_type = 'text'
        if isinstance(current_msg_val, dict):
            # Try to find 'type' in the top level
            msg_type = current_msg_val.get('type', 'text')
            
            # If not found, sometimes it's nested in 'data' (rare but possible)
            if 'data' in current_msg_val and isinstance(current_msg_val['data'], dict):
                 if 'type' in current_msg_val['data']:
                     msg_type = current_msg_val['data']['type']

        logger.info(f"🔍 Detected Message Type: {msg_type}")

        violation_found = False
        reason_msg = ""

        # ==========================================
        # 📷 OPTION A: IMAGE MODERATION
        # ==========================================
        if msg_type == 'image' and isinstance(current_msg_val, dict):
            # Extract URL
            image_url = current_msg_val.get('data', {}).get('url')
            
            # Fallback: Sometimes URL is inside 'entities' or other fields
            if not image_url and 'data' in current_msg_val:
                 # Print structure to debug if URL is missing
                 # logger.info(f"Debug Image Data: {current_msg_val['data']}") 
                 pass

            if image_url:
                logger.info(f"📷 Analyzing Image URL: {image_url}")
                response = requests.get(image_url)
                
                if response.status_code == 200:
                    try:
                        # Send raw bytes to Azure
                        request_options = AnalyzeImageOptions(image=ImageData(content=response.content))
                        result = client.analyze_image(request_options)
                        
                        # STRICT Rule for Images (Level 2 blocks Nudity)
                        violation_found, reason_msg = check_safety(result, block_level=2)
                        
                    except HttpResponseError as e:
                        logger.error(f"Azure Error: {e}")
                else:
                    logger.warning("Could not download image")
            else:
                 logger.warning("⚠️ Received 'image' type but could not find 'url' in data.")

        # ==========================================
        # 💬 OPTION B: TEXT MODERATION
        # ==========================================
        else:
            combined_text = ""
            # Loop through history safely
            for entry in context_list:
                val = next(iter(entry.values()))
                if isinstance(val, str):
                    combined_text += f"{val}. "
                elif isinstance(val, dict):
                    # Try to find text in 'data' -> 'text'
                    text_part = val.get('data', {}).get('text', '')
                    if text_part:
                        combined_text += f"{text_part}. "
                    
                    # If that's empty, maybe it's a file name? (Don't analyze file metadata as text)
                    elif val.get('type') == 'file' or val.get('type') == 'image':
                         continue # Skip metadata for images/files in text analysis

            if combined_text.strip():
                logger.info(f"🧐 Analyzing Text: {combined_text[:50]}...")
                try:
                    request_options = AnalyzeTextOptions(text=combined_text[:1000])
                    result = client.analyze_text(request_options)
                    
                    # STANDARD Rule for Text (Level 4 blocks Hate/Violence)
                    violation_found, reason_msg = check_safety(result, block_level=4)
                    
                except HttpResponseError as e:
                    logger.error(f"Azure Error: {e}")

        # ==========================================
        # 🏁 RESULT
        # ==========================================
        logger.info(f"⏱️ Time: {time.time() - start_time:.2f}s")

        if violation_found:
            logger.warning(f"🚫 BLOCKED: {reason_msg}")
            return {"isMatchingCondition": True, "confidence": 100, "reason": reason_msg}
        
        logger.info("✅ Safe")
        return {"isMatchingCondition": False, "confidence": 0, "reason": "Safe"}

    except Exception as e:
        logger.error(f"❌ Critical Error: {e}")
        return {"isMatchingCondition": False}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)