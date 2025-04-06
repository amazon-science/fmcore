import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages.base import BaseMessageChunk

from fmcore.llm.types.llm_types import LLMConfig, ModelParams
from fmcore.llm.types.provider_types import BedrockProviderParams, BedrockAccountConfig
from fmcore.types.enums.provider_enums import ProviderType
from fmcore.aws.factory.bedrock_factory import BedrockClientProxy
from aiolimiter import AsyncLimiter

@pytest.fixture
def mock_bedrock_client():
    """Create a mock Bedrock client with all necessary methods."""
    client = MagicMock()
    client.invoke = MagicMock(return_value=AIMessage(content="Test response"))
    client.ainvoke = AsyncMock(return_value=AIMessage(content="Test async response"))
    client.stream = MagicMock(return_value=[BaseMessageChunk(content="Test", type="ai")])
    client.astream = AsyncMock(return_value=[BaseMessageChunk(content="Test", type="ai")])
    return client

@pytest.fixture
def mock_bedrock_client_proxy(mock_bedrock_client):
    """Create a mock BedrockClientProxy with rate limiting."""
    rate_limiter = AsyncLimiter(max_rate=10)
    return BedrockClientProxy(client=mock_bedrock_client, rate_limiter=rate_limiter)

@pytest.fixture
def bedrock_llm_config():
    """Create a test Bedrock LLM configuration."""
    return LLMConfig(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        provider_params=BedrockProviderParams(
            provider_type=ProviderType.BEDROCK,
            accounts=[
                BedrockAccountConfig(
                    region="us-east-1",
                    role_arn="arn:aws:iam::123456789012:role/test-role",
                    rate_limit=50
                ),
                BedrockAccountConfig(
                    region="us-west-2",
                    role_arn="arn:aws:iam::123456789012:role/test-role-2",
                    rate_limit=100
                )
            ]
        ),
        model_params=ModelParams(
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9
        )
    )

@pytest.fixture
def test_messages():
    """Create a list of test messages."""
    return [
        HumanMessage(content="Hello, how are you?"),
        AIMessage(content="I'm doing well, thank you!"),
        HumanMessage(content="What's the weather like?")
    ] 