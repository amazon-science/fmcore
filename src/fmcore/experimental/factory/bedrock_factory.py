from typing import List, TypeAlias

from aiolimiter import AsyncLimiter
from langchain_aws import ChatBedrockConverse

from fmcore.experimental.factory.boto_factory import BotoFactory
from fmcore.experimental.types.llm_types import LLMConfig
from fmcore.experimental.proxy.rate_limit_proxy import RateLimitedProxy
from fmcore.experimental.types.provider_types import BedrockAccountConfig

# Clearer alias
BedrockClientProxy: TypeAlias = RateLimitedProxy[ChatBedrockConverse]


class BedrockFactory:
    """Factory class for creating Bedrock clients with additional functionalities like rate limiting."""

    @staticmethod
    def create_bedrock_clients(*, llm_config: LLMConfig) -> List[BedrockClientProxy]:
        """Creates multiple Bedrock clients based on the provided configuration."""
        return [
            BedrockFactory._create_bedrock_client_with_converse(account, llm_config)
            for account in llm_config.provider_params.accounts
        ]

    @staticmethod
    def _create_bedrock_client_with_converse(
            account_config: BedrockAccountConfig, llm_config: LLMConfig
    ) -> BedrockClientProxy:
        """Helper method to create a single Bedrock client with rate limiting."""
        boto_client = BotoFactory.get_client(
            service_name="bedrock-runtime",
            region=account_config.region,
            role_arn=account_config.role_arn,
        )

        converse_client = ChatBedrockConverse(
            model_id=llm_config.model_id,
            client=boto_client,
            **llm_config.model_params.dict(exclude_none=True),
        )

        # Currently, We are using the off the shelf rate limiters provided by aiolimiter
        # TODO Implement custom rate limiters
        rate_limiter = AsyncLimiter(max_rate=account_config.rate_limit)

        return BedrockClientProxy(client=converse_client, rate_limiter=rate_limiter)
