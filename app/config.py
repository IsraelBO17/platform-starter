# app/config.py
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every value comes from the environment or a local .env file. Nothing is hard-coded."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Which provider and model the platform uses. Change these, not the code.
    llm_provider: str = "groq"            # groq | gemini | openai | xai | ollama | anthropic | bedrock
    llm_model: str = "gemini-3.8-flash"
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 4

    # Keys. Only the one for your chosen provider needs a value.
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    xai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # AWS (Bedrock, Polly, Transcribe). boto3 reads credentials from the standard
    # AWS environment variables or profile; we only need the Region here.
    aws_region: str = "us-east-1"

    # Callers must send this in an X-API-Key header. Empty = no check (local development only).
    app_api_key: str | None = None

    database_url: str = "postgresql://postgres:postgres@localhost:5432/platform"


@lru_cache
def get_settings() -> Settings:
    from dotenv import load_dotenv

    from app.secrets import load_secrets_from_ssm

    load_dotenv()             # also exposes .env to boto3 and other libraries (AWS_PROFILE, HF_TOKEN)
    load_secrets_from_ssm()   # no-op locally; fills keys from SSM on Lambda
    return Settings()
