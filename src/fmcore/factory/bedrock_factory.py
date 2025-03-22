from typing import List, TypeAlias

from aiolimiter import AsyncLimiter
from langchain_aws import ChatBedrockConverse

from fmcore.factory.boto_factory import BotoFactory
from fmcore.types.llm_types import LLMConfig
from fmcore.proxy.rate_limit_proxy import RateLimitedProxy
from fmcore.types.provider_types import BedrockAccountConfig

BedrockClientProxy: TypeAlias = RateLimitedProxy[ChatBedrockConverse]


class BedrockFactory:
    """Factory class for creating Bedrock clients with additional functionalities like rate limiting.

    This class provides static methods to create and configure Amazon Bedrock clients
    with built-in rate limiting capabilities. It handles the creation of both single
    and multiple clients based on provided configurations.

    The factory supports multiple AWS accounts and automatically configures rate limiting
    based on account-specific parameters.
    """

    @staticmethod
    def create_bedrock_clients(llm_config: LLMConfig) -> List[BedrockClientProxy]:
        """Creates multiple Bedrock clients based on the provided configuration.

        Args:
            llm_config (LLMConfig): Configuration object containing LLM settings and provider parameters,
                                  including account configurations and model parameters.

        Returns:
            List[BedrockClientProxy]: A list of rate-limited Bedrock client proxies, one for each
                                    account specified in the configuration.

        Example:
            llm_config = LLMConfig(...)
            clients = BedrockFactory.create_bedrock_clients(llm_config)
        """
        return [
            BedrockFactory._create_bedrock_client_with_converse(
                account_config=account, llm_config=llm_config
            )
            for account in llm_config.provider_params.accounts
        ]

    @staticmethod
    def _create_bedrock_client_with_converse(
        account_config: BedrockAccountConfig, llm_config: LLMConfig
    ) -> BedrockClientProxy:
        """Creates a single Bedrock client with rate limiting capabilities.

        Args:
            account_config (BedrockAccountConfig): Configuration for a specific AWS account,
                                                 including region, role ARN, and rate limits.
            llm_config (LLMConfig): Configuration containing model settings and parameters.

        Returns:
            BedrockClientProxy: A rate-limited proxy wrapper around the Bedrock client.

        Note:
            The method configures rate limiting based on the account's specified rate limit
            and wraps the ChatBedrockConverse client in a proxy for controlled access.
        """
        boto_client = BotoFactory.get_client(
            service_name="bedrock-runtime",
            region=account_config.region,
            role_arn=account_config.role_arn,
        )

        converse_client = ChatBedrockConverse(
            model_id=llm_config.model_id,
            client=boto_client,
            **llm_config.model_params.model_dump(exclude_none=True),
        )

        # Create rate limiter based on account config
        rate_limiter = AsyncLimiter(max_rate=account_config.rate_limit)

        # Create proxy without weight
        return BedrockClientProxy(client=converse_client, rate_limiter=rate_limiter)
