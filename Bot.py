import logging
import os
import requests
import json
import asyncio
import base64
from dotenv import load_dotenv
from io import BytesIO

from telegram import Update, constants
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Load environment variables from .env file
load_dotenv()

# Get token and key from environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"

# Set up logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Function to call the Gemini API with text ---
async def call_gemini_api(prompt: str) -> str:
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.post(GEMINI_API_URL, headers=headers, json=data, timeout=60))
        response.raise_for_status()
        response_json = response.json()

        if "candidates" in response_json and len(response_json["candidates"]) > 0:
            candidate = response_json["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"] and len(candidate["content"]["parts"]) > 0:
                return candidate["content"]["parts"][0].get("text", "No text found in the response.")
            elif "finishReason" in candidate:
                return f"Generation stopped: {candidate['finishReason']}. Try a different question."
            else:
                return "Failed to process the response from the AI (unexpected structure)."
        elif "promptFeedback" in response_json and "blockReason" in response_json["promptFeedback"]:
            return f"Request blocked: {response_json['promptFeedback']['blockReason']}."
        else:
            return "Failed to get a valid response from the AI."

    except Exception as e:
        logger.error(f"Error during request to Gemini API: {e}")
        return f"Sorry, an error occurred: {e}"

# --- Function to call Gemini API with an image ---
async def call_gemini_with_image(prompt: str, image_bytes: bytes) -> str:
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    data = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": base64_image
                    }
                }
            ]
        }]
    }

    headers = {'Content-Type': 'application/json'}

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.post(GEMINI_API_URL, headers=headers, json=data, timeout=60))
        response.raise_for_status()
        response_json = response.json()

        if "candidates" in response_json and response_json["candidates"]:
            return response_json["candidates"][0]["content"]["parts"][0].get("text", "No response text found.")
        else:
            return "Couldn't get a valid response from the AI."

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return f"An error occurred: {e}"

# --- Command: /start ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_html(
        rf"Hello {user.mention_html()}! Send me a question or a photo, and I'll respond using Gemini AI."
    )

# --- Command: /help ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send me:\n"
        "- Any text (e.g., 'Explain quantum physics')\n"
        "- A photo (I'll describe it or answer questions about it)\n\n"
        "I'm powered by Gemini AI!"
    )

# --- Handler for text messages ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    chat_id = update.effective_chat.id
    logger.info(f"Text from {update.effective_user.username}: {user_message}")

    await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
    ai_response = await call_gemini_api(user_message)

    try:
        await update.message.reply_text(ai_response)
    except Exception as e:
        logger.error(f"Failed to send reply message: {e}")
        await update.message.reply_text("An error occurred while sending the reply.")

# --- Handler for image messages ---
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    photos = update.message.photo

    if not photos:
        await update.message.reply_text("No image found.")
        return

    photo_file = await photos[-1].get_file()
    photo_bytes = await photo_file.download_as_bytearray()

    prompt = "What is in this image?"
    await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
    ai_response = await call_gemini_with_image(prompt, photo_bytes)

    await update.message.reply_text(ai_response)

# --- Main bot setup ---
def main() -> None:
    if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
        logger.error("Missing TELEGRAM_BOT_TOKEN or GEMINI_API_KEY in environment variables.")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    logger.info("Bot starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Bot stopped.")

# --- Run the bot ---
if __name__ == "__main__":
    main()