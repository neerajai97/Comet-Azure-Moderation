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
    """Check if any category exceeds the severity threshold."""
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

        # Get the current message (last in the array)
        last_entry = context_list[-1]
        current_msg_val = next(iter(last_entry.values()))

        # Determine message type from the CURRENT message
        msg_type = 'text'
        if isinstance(current_msg_val, dict):
            msg_type = current_msg_val.get('type', 'text')

        logger.info(f"🔍 Message Type: {msg_type} | Context Window: {len(context_list)} messages")

        violation_found = False
        reason_msg = ""

        # ==========================================
        # 📷 IMAGE MODERATION (Current Image Only)
        # ==========================================
        if msg_type == 'image' and isinstance(current_msg_val, dict):
            image_url = current_msg_val.get('data', {}).get('url')

            if image_url:
                logger.info(f"📷 Analyzing Image: {image_url}")
                try:
                    response = requests.get(image_url, timeout=10)
                    
                    if response.status_code == 200:
                        # Analyze the actual image content
                        request_options = AnalyzeImageOptions(image=ImageData(content=response.content))
                        result = client.analyze_image(request_options)
                        violation_found, reason_msg = check_safety(result, block_level=2)
                    else:
                        logger.warning(f"Could not download image (HTTP {response.status_code})")
                        
                except Exception as e:
                    logger.error(f"Image download/analysis error: {e}")
            else:
                logger.warning("⚠️ Image type detected but no URL found")

            # RETURN immediately after image analysis
            logger.info(f"⏱️ Processing Time: {time.time() - start_time:.2f}s")
            
            if violation_found:
                logger.warning(f"🚫 BLOCKED IMAGE: {reason_msg}")
                return {
                    "isMatchingCondition": True,
                    "confidence": 0.95,
                    "reason": reason_msg
                }
            
            logger.info("✅ Image Safe")
            return {
                "isMatchingCondition": False,
                "confidence": 0.98,
                "reason": "Image is safe"
            }

        # ==========================================
        # 💬 TEXT MODERATION (All Context Messages)
        # ==========================================
        else:
            combined_text = ""
            text_messages_count = 0
            
            # Extract text from ALL messages in the context window
            for entry in context_list:
                val = next(iter(entry.values()))
                
                # Handle string messages (legacy format)
                if isinstance(val, str):
                    combined_text += val + "\n"
                    text_messages_count += 1
                
                # Handle structured messages
                elif isinstance(val, dict):
                    entry_type = val.get('type', 'text')
                    
                    # CRITICAL FIX: Only extract text from text-type messages
                    # Skip images, files, audio, video, etc.
                    if entry_type == 'text':
                        # Look for text in the 'data' object
                        data_obj = val.get('data', {})
                        
                        # Handle both string and object formats
                        if isinstance(data_obj, str):
                            text_content = data_obj
                        elif isinstance(data_obj, dict):
                            text_content = data_obj.get('text', '')
                        else:
                            text_content = ''
                        
                        if text_content and text_content.strip():
                            combined_text += text_content + "\n"
                            text_messages_count += 1

            if combined_text.strip():
                logger.info(f"🧐 Analyzing {text_messages_count} text messages from context")
                logger.info(f"📝 Combined Text Preview: {combined_text[:100]}...")
                
                try:
                    # Azure has a 10,000 character limit for text analysis
                    truncated_text = combined_text[:10000]
                    
                    if len(combined_text) > 10000:
                        logger.warning(f"⚠️ Text truncated from {len(combined_text)} to 10,000 chars")
                    
                    request_options = AnalyzeTextOptions(text=truncated_text)
                    result = client.analyze_text(request_options)
                    violation_found, reason_msg = check_safety(result, block_level=2)
                    
                except HttpResponseError as e:
                    logger.error(f"Azure Text Analysis Error: {e}")
            else:
                logger.info("ℹ️ No text content found in context messages")

            # RETURN for text analysis
            logger.info(f"⏱️ Processing Time: {time.time() - start_time:.2f}s")
            
            if violation_found:
                logger.warning(f"🚫 BLOCKED TEXT: {reason_msg}")
                return {
                    "isMatchingCondition": True,
                    "confidence": 0.95,
                    "reason": reason_msg
                }
            
            logger.info("✅ Text Safe")
            return {
                "isMatchingCondition": False,
                "confidence": 0.98,
                "reason": "Content is safe"
            }

    except Exception as e:
        logger.error(f"❌ Critical Error: {e}", exc_info=True)
        # Fail open to prevent blocking good users on server errors
        return {"isMatchingCondition": False}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)