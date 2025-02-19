# Telegram Anti-Cyberbullying Bot 🤖🚫💬

This is a Telegram bot designed to promote healthy interactions in group chats by detecting and removing offensive or abusive messages. The bot automatically analyzes messages in real-time and enforces moderation by removing users who violate group guidelines.

---

## Features
- 🌟 **Real-Time Moderation**: The bot monitors all messages in the group and flags offensive content.
- ⚠️ **Warning System**: Warns users for their first offense (if enabled).
- ❌ **Automatic Removal**: Removes users from the group upon detecting repeated violations or highly offensive content.
- 🧠 **AI-Powered**: Uses Natural Language Processing (NLP) models to detect offensive language.
- 🛡️ **Customizable Rules**: Admins can configure offensive thresholds and manage ban settings.
- 📊 **Logs**: Keeps a record of removed users and flagged messages for admin review.

---

## How It Works
1. **Message Monitoring**: The bot listens to every message in the group.
2. **Offensive Content Detection**:
   - Offensive language is detected using a machine learning model, such as a pretrained toxicity classifier (e.g., OpenAI, Hugging Face, or custom-trained models).
   - Messages are scored based on their offensiveness.
3. **Action Execution**:
   - If the message surpasses the offensive threshold, the bot warns the user or removes them from the group.
   - Logs the message and action taken for admins.

---

## Requirements
- Python 3.8+
- Telegram Bot Token (Get it from [BotFather](https://core.telegram.org/bots#botfather))
- Pretrained toxicity detection model (e.g., Hugging Face transformers or OpenAI Whisper for speech-to-text analysis)
- [FFmpeg](https://ffmpeg.org/download.html) (if using audio-to-text features)

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Telegram-Anti-Cyberbullying-Bot.git
   cd Telegram-Anti-Cyberbullying-Bot
