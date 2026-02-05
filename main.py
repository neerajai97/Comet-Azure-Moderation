import uvicorn
import logging
from fastapi import FastAPI, Request
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.core.credentials import AzureKeyCredential
import time
from dotenv import load_dotenv
import os

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

@app.post("/webhook")
async def handle_webhook(request: Request):
    """
    Handles the CometChat 'Custom API' request.
    Expected Payload: { "contextMessages": [ ... ] }
    Expected Response: { "isMatchingCondition": bool, "reason": str }
    """
    start_time = time.time()
    try:
        payload = await request.json()
        
        # 1. PARSE CONTEXT (The Custom API structure)
        # The Custom API sends a list of messages (history + current)
        context_list = payload.get('contextMessages', [])
        
        if not context_list:
            # If empty, allow the message
            return {"isMatchingCondition": False}

        # 2. COMBINE TEXT
        # We assume the LAST item is the current message (complex object)
        # The previous items are history (simple text)
        
        combined_text = ""
        
        # Separate history vs current
        # (The logic here handles both simple key-value pairs and complex objects)
        for entry in context_list:
            for key, value in entry.items():
                if isinstance(value, str):
                    # History message: {"uid": "text"}
                    combined_text += f"{value}."
                elif isinstance(value, dict) and 'data' in value:
                    # Current message object
                    current_text = value['data'].get('text', '')
                    combined_text += f"{current_text}"

        logger.info(f"🧐 Analyzing: {combined_text}")

        # 3. ASK AZURE TO ANALYZE
        # Truncate to 1000 chars to avoid Azure limits
        request_options = AnalyzeTextOptions(text=combined_text[:1000])
        
        # Sync call inside async function (FastAPI handles this fine, or use await if async SDK available)
        result = client.analyze_text(request_options)

        # 4. CHECK SEVERITY
        # Severity 4+ is usually the blocking threshold
        BLOCK_LEVEL = 2
        violation_found = False
        reason_msg = ""

        for category in result.categories_analysis:
            if category.severity >= BLOCK_LEVEL:
                violation_found = True
                reason_msg = f"{category.category} (Level {category.severity})"
                break

        duration = time.time() - start_time
        logger.info(f"⏱️ Processing time: {duration:.2f}s")

        # 5. RETURN THE *EXACT* JSON COMETCHAT WANTS
        if violation_found:
            logger.warning(f"🚫 BLOCKED: {reason_msg}")
            return {
                "isMatchingCondition": True,
                "confidence": 0.95,
                "reason": reason_msg
            }
        
        # If Safe
        logger.info("✅ Safe. Allowed.")
        return {
            "isMatchingCondition": False,
            "confidence": 0.98,
            "reason": "Content is safe"
        }

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        # FAIL OPEN: If code crashes, allow message to avoid service outage
        return {"isMatchingCondition": False}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)