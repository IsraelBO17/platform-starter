# app/providers/factory.py
from app.config import Settings
from app.providers.base import LLMProvider
from app.providers.openai_compatible import BASE_URLS, OpenAICompatibleProvider


def build_provider(settings: Settings) -> LLMProvider:
    """Pick the provider named in configuration. The rest of the code never knows which."""
    name = settings.llm_provider
    if name == "bedrock":
        from app.providers.bedrock import BedrockProvider  # imported only when used

        return BedrockProvider(region=settings.aws_region, timeout=settings.llm_timeout_seconds)
    if name in BASE_URLS:
        key = getattr(settings, f"{name}_api_key", None)
        if name != "ollama" and not key:
            raise RuntimeError(f"Set {name.upper()}_API_KEY to use the {name} provider.")
        return OpenAICompatibleProvider(name, key, settings.llm_timeout_seconds)
    raise RuntimeError(f"Unsupported LLM_PROVIDER: {name}")
