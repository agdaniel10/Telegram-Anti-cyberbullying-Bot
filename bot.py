import os
import logging
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import (
    filters,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    Application,
)
from transformers import GPT2Tokenizer
import torch
import torch.nn.functional as F
from db import add_offense, get_offense_count, reset_offense
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# Download the model file from Hugging Face
model_path = hf_hub_download(
    repo_id="agdaniel10/Cyberbullying_Model",  # Your repository name
    filename="tg_cyberbullying_model.pth",     # Your model file name
)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)

# Set the model to evaluation mode
model.eval()

print("Model loaded successfully!")

# Initialize tokenizer and add special token
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = '[PAD]'

# Preprocess input sentence for tokenization
def preprocess_input(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    return inputs

# Test sentence using the model
def test_sentence(sentence):
    if model is None:
        logging.error("Model is not loaded. Cannot process input.")
        return None, None

    # Preprocess the input sentence
    inputs = preprocess_input(sentence)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs, dim=-1)  # Apply softmax to logits
        predicted_class = torch.argmax(probabilities, dim=1).item()  # Get the predicted class

    return predicted_class, probabilities.cpu()

# Load the environment variables
load_dotenv()

# Bot token and username (from environment variable for security)
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError('Bot token not found. Set the TELEGRAM_BOT_TOKEN environment variable.')

BOT_USERNAME = '@anti_abuse_and_cyberbully_bot'

# Configure logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

# Command handler for the /start command
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=("This bot is against Cyberbullying. "
              "I will detect and remove those who engage in cyberbullying and abuse in group chats.")
    )

# Constants
MAXIMUM_BULLY_MESSAGES = 8
MAXIMUM_DAYS_BANNED = 1

# Cyberbullying Handler Function
async def remove_cyberbullying(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    message = update.message

    if not message or not message.from_user:
        logging.info("Update does not contain a valid message or sender. Skipping.")
        return

    sender_id = message.from_user.id

    if message.text == BOT_USERNAME:
        await context.bot.send_message(
            chat_id=chat_id,
            text="This bot is against Cyberbullying. I will detect and remove those who engage in cyberbullying and abuse in group chats."
        )
        return

    # Log the received message
    if message.text:
        logging.info(f"Message received: {message.text} from user: {sender_id}")
    else:
        logging.info("Non-text message received. Skipping.")
        return

    # Use the model to classify the message
    try:
        predicted_class, probabilities = test_sentence(message.text)
    except Exception as e:
        logging.error(f"Error during model inference: {e}")
        return

    # If the message is classified as non-cyberbullying (class 0)
    if predicted_class == 0:
        logging.info("Message classified as non-cyberbullying. Skipping.")
        return
    
    # Otherwise, it's bullying (class 1)
    logging.info("Message classified as cyberbullying.")

    # Track the offense in the database
    try:
        add_offense(sender_id, chat_id, message.text)
    except Exception as e:
        logging.error(f"Failed to update offense count for user {sender_id}: {e}")
        return

    # Check if the user has reached the bullying threshold
    try:
        if get_offense_count(sender_id, chat_id) >= MAXIMUM_BULLY_MESSAGES:
            current_date = datetime.now()
            ban_duration = timedelta(days=MAXIMUM_DAYS_BANNED)
            unban_date = current_date + ban_duration

            # Reset the user's record in the database
            reset_offense(sender_id, chat_id)

            # Notify the user and ban them
            await context.bot.send_message(
                chat_id=chat_id,
                text=f'You have been banned for {MAXIMUM_DAYS_BANNED} days due to repeated offensive messages.',
                reply_to_message_id=message.message_id
            )

            try:
                await context.bot.ban_chat_member(
                    chat_id=chat_id,
                    user_id=sender_id,
                    until_date=unban_date.timestamp(),
                    revoke_messages=False
                )
                logging.info(f"User {sender_id} banned from chat {chat_id} until {unban_date}.")
            except Exception as e:
                logging.error(f"Failed to ban user {sender_id} in chat {chat_id}: {e}")
        else:
            # Notify the user of the warning
            await context.bot.send_message(
                chat_id=chat_id,
                text=f'The message you sent is abusive and is not permitted in this group.',
                reply_to_message_id=message.message_id
            )
    except Exception as e:
        logging.error(f"Error handling threshold or ban for user {sender_id}: {e}")

# Main function to start the bot
def main():
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, remove_cyberbullying))

    application.run_polling()

if __name__ == "__main__":
    main()