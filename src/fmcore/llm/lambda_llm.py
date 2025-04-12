import json
from typing import List, Iterator, AsyncIterator, Dict

import aioboto3
from aiolimiter import AsyncLimiter
from botocore.client import BaseClient
from langchain_aws import ChatBedrockConverse
from langchain_community.adapters.openai import convert_dict_to_message
from pydantic import BaseModel
from langchain_core.messages import (
    BaseMessage,
    BaseMessageChunk,
    messages_to_dict,
    convert_to_openai_messages,
)

from fmcore.aws.factory.bedrock_factory import BedrockFactory
from fmcore.aws.factory.boto_factory import BotoFactory
from fmcore.llm.base_llm import BaseLLM
from fmcore.llm.types.llm_types import LLMConfig
from fmcore.llm.types.provider_types import LambdaProviderParams
from fmcore.utils.rate_limit_utils import RateLimiterUtils
from fmcore.utils.retry_utils import RetryUtil


class LambdaLLM(BaseLLM[List[BaseMessage], BaseMessage, BaseMessageChunk], BaseModel):
    """
    An LLM implementation that routes requests through an AWS Lambda function.

    This class uses both synchronous and asynchronous boto3 Lambda clients to
    interact with an LLM hosted via AWS Lambda. It includes automatic async rate
    limiting and supports OpenAI-style message formatting.

    Attributes:
        sync_client (BaseClient): Boto3 synchronous client for AWS Lambda.
        async_session ( aioboto3.Session): Boto3 asynchronous session for AWS Lambda.
        rate_limiter (AsyncLimiter): Async limiter to enforce API rate limits.

    Note:
        The `async_client` is not stored directly because `aioboto3.client(...)` returns
        an asynchronous context manager, which must be used with `async with` and cannot
        be reused safely across calls. Instead, we store an `aioboto3.Session` instance
        in `async_session`, from which a fresh client is created inside each `async with`
        block

    """

    aliases = ["LAMBDA"]

    sync_client: BaseClient
    async_session: aioboto3.Session  # Using session here as aioboto3.client returns context manager
    rate_limiter: AsyncLimiter

    @classmethod
    def _get_instance(cls, *, llm_config: LLMConfig) -> "LambdaLLM":
        """
        Factory method to create an instance of LambdaLLM with the given configuration.

        Args:
            llm_config (LLMConfig): The LLM configuration, including model and provider details.

        Returns:
            LambdaLLM: A configured instance of the Lambda-backed LLM.
        """
        provider_params: LambdaProviderParams = llm_config.provider_params

        sync_client = BotoFactory.get_client(
            service_name="lambda",
            region=provider_params.region,
            role_arn=provider_params.role_arn,
        )
        async_session = BotoFactory.get_async_session(
            service_name="lambda",
            region=provider_params.region,
            role_arn=provider_params.role_arn,
        )

        rate_limiter = RateLimiterUtils.create_async_rate_limiter(
            rate_limit_config=provider_params.rate_limit
        )

        return LambdaLLM(
            config=llm_config, sync_client=sync_client, async_session=async_session, rate_limiter=rate_limiter
        )

    def convert_messages_to_lambda_payload(self, messages: List[BaseMessage]) -> Dict:
        """
        Converts internal message objects to the payload format expected by the Lambda function.
        We expect all lambdas to be accepting openai messages format

        Args:
            messages (List[BaseMessage]): List of internal message objects.

        Returns:
            Dict: The payload dictionary to send to the Lambda function.
        """
        return {
            "modelId": self.config.model_id,
            "messages": convert_to_openai_messages(messages),
            "model_params": self.config.model_params.model_dump(),
        }

    def convert_lambda_response_to_messages(self, response: Dict) -> BaseMessage:
        """
        Converts the raw Lambda function response into a BaseMessage.

        This method expects the Lambda response to contain a 'Payload' key with a stream
        of OpenAI-style messages (a list of dictionaries). It parses the stream, extracts
        the first message, and converts it into a BaseMessage instance.

        Args:
            response (Dict): The response dictionary returned from the Lambda invocation.

        Returns:
            BaseMessage: The first parsed message from the response.
        """
        response_payload: List[Dict] = json.load(response["Payload"])
        # The Lambda returns a list of messages in OpenAI format.
        # Currently, we only expect a single response message,
        # so we take the first item in the list.
        return convert_dict_to_message(response_payload[0])

    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """
        Synchronously invokes the Lambda function with given messages.

        Args:
            messages (List[BaseMessage]): Input messages for the model.

        Returns:
            BaseMessage: Response message from the model.
        """
        payload = self.convert_messages_to_lambda_payload(messages)
        response = self.sync_client.invoke(
            FunctionName=self.config.provider_params.function_arn,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload),
        )
        return self.convert_lambda_response_to_messages(response)

    @RetryUtil.with_backoff(lambda self: self.config.provider_params.retries)
    async def ainvoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """
        Asynchronously invokes the Lambda function with rate limiting.

        Args:
            messages (List[BaseMessage]): Input messages for the model.

        Returns:
            BaseMessage: Response message from the model.
        """
        async with self.rate_limiter:
            async with self.async_session.client("lambda", region_name="us-west-2") as lambda_client:
                payload = self.convert_messages_to_lambda_payload(messages)
                response = await lambda_client.invoke(
                    FunctionName=self.config.provider_params.function_arn,
                    InvocationType="RequestResponse",
                    Payload=json.dumps(payload),
                )
                payload = await response["Payload"].read()
                response_payload: List[Dict] = json.loads(payload.decode("utf-8"))
                # The Lambda returns a list of messages in OpenAI format.
                # Currently, we only expect a single response message,
                # so we take the first item in the list.
                return convert_dict_to_message(response_payload[0])
                #return self.convert_lambda_response_to_messages(response)

    def stream(self, messages: List[BaseMessage]) -> Iterator[BaseMessageChunk]:
        """
        Not implemented. Streaming is not supported for LambdaLLM.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError("Streaming is not implemented for LambdaLLM")

    async def astream(self, messages: List[BaseMessage]) -> AsyncIterator[BaseMessageChunk]:
        """
        Not implemented. Asynchronous streaming is not supported for LambdaLLM.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError("Streaming is not implemented for LambdaLLM")

    def batch(self, messages: List[List[BaseMessage]]) -> List[BaseMessage]:
        """
        Not implemented. Batch processing is not supported for LambdaLLM.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError("Batch processing is not implemented for LambdaLLM.")

    async def abatch(self, messages: List[List[BaseMessage]]) -> List[BaseMessage]:
        """
        Not implemented. Asynchronous batch processing is not supported for LambdaLLM.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError("Batch processing is not implemented for LambdaLLM.")
