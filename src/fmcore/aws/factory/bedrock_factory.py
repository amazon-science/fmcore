from langchain_aws import ChatBedrockConverse

from fmcore.aws.factory.boto_factory import BotoFactory
from fmcore.llm.types.llm_types import LLMConfig
from fmcore.llm.types.provider_types import BedrockProviderParams


class BedrockFactory:
    """
    Factory class for creating Bedrock conversation clients.

    This class provides static methods to construct Bedrock clients using the provided
    LLM configuration. It focuses solely on client creation, delegating any rate limiting
    concerns to separate utilities if needed.
    """

    @staticmethod
    def create_converse_client(llm_config: LLMConfig) -> ChatBedrockConverse:
        """
        Creates a ChatBedrockConverse client for a single-account configuration.

        This method instantiates a ChatBedrockConverse client using the LLM configuration,
        which includes global model parameters and provider-specific settings such as AWS account
        details (region and role ARN). It uses BotoFactory to obtain a boto client for the
        'bedrock-runtime' service and then creates the conversation client with the specified
        model and parameters.

        Args:
            llm_config (LLMConfig): Configuration containing:
                - model_id: A unique identifier for the model.
                - model_params: Global model parameters (e.g., temperature, max_tokens, top_p).
                - provider_params: Provider-specific settings. Must be an instance of BedrockProviderParams,
                  which includes AWS region and role ARN.

        Returns:
            ChatBedrockConverse: A fully configured conversation client ready for use.

        Raises:
            AssertionError: If llm_config.provider_params is not an instance of BedrockProviderParams.
        """
        # Ensure provider_params is of type BedrockProviderParams.
        assert isinstance(llm_config.provider_params, BedrockProviderParams), (
            "Expected provider_params to be an instance of BedrockProviderParams"
        )
        provider_params: BedrockProviderParams = llm_config.provider_params

        boto_client = BotoFactory.get_client(
            service_name="bedrock-runtime",
            region=provider_params.region,
            role_arn=provider_params.role_arn,
        )

        converse_client = ChatBedrockConverse(
            model_id=llm_config.model_id,
            client=boto_client,
            **llm_config.model_params.model_dump(exclude_none=True),
        )

        return converse_client
