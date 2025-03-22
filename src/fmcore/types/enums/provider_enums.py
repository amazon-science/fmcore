from enum import Enum


class ProviderType(str, Enum):
    """Enum representing different provider types."""

    BEDROCK = "BEDROCK"
    LAMBDA = "LAMBDA"
    OPENAI = "OPENAI"
