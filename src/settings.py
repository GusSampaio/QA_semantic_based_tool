import os
from dotenv import load_dotenv

load_dotenv()

class AppSettings:
    def __init__(self):
        self.mistral_api_key: str | None = os.getenv("MISTRAL_API_KEY")
        self.chat_model: str = os.getenv("CHAT_MODEL", "mistral-small-latest")


def missing_required_env(settings: AppSettings) -> bool:
    return not bool(settings.mistral_api_key)