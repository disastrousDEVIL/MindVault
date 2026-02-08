import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    # App
    APP_NAME = "MindVault"
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"


settings = Settings()