import json
import random
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Union

from autoenum import AutoEnum, auto
from bears import FileMetadata
from bears.constants import Parallelize
from bears.util import (
    Alias,
    Log,
    Parameters,
    String,
    accumulate,
    any_are_none,
    as_list,
    dispatch,
    dispatch_executor,
    optional_dependency,
    remove_values,
    retry,
    set_param_from_alias,
    stop_executor,
)
from bears.util.aws.iam import IAMUtil
from pydantic import confloat, conint, constr, model_validator

from fmcore.framework._task.text_generation import (
    GENERATED_TEXTS_COL,
    GenerativeLM,
    Prompts,
    TextGenerationParams,
    TextGenerationParamsMapper,
)


class ConfigSelectionStrategy(AutoEnum):
    RANDOM_AT_INIT = auto()
    RANDOM_PER_REQUEST = auto()
    ROUND_ROBIN_AT_INIT = auto()


with optional_dependency("boto3"):
    from botocore.exceptions import ClientError

    def call_claude_v1_v2(
        bedrock_client,
        model_name: str,
        prompt: str,
        max_tokens_to_sample: int,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        assert any_are_none(top_k, top_p), "At least one of top_k, top_p must be None"
        bedrock_params = {
            "prompt": prompt,
            "max_tokens_to_sample": max_tokens_to_sample,
        }
        if top_p is not None and temperature is not None:
            raise ValueError("Cannot specify both top_p and temperature; at most one must be specified.")

        if top_k is not None:
            assert isinstance(top_k, int)
            bedrock_params["top_k"] = top_k
        elif temperature is not None:
            assert isinstance(temperature, (float, int)) and 0 <= temperature <= 1
            bedrock_params["temperature"] = temperature
        elif top_p is not None:
            assert isinstance(top_p, (float, int)) and 0 <= top_p <= 1
            bedrock_params["top_p"] = top_p

        if stop_sequences is not None:
            bedrock_params["stop_sequences"] = stop_sequences

        response = bedrock_client.invoke_model(
            body=json.dumps(bedrock_params),
            modelId=model_name,
            accept="application/json",
            contentType="application/json",
        )
        response_body: Dict = json.loads(response.get("body").read())
        return response_body.get("completion")

    def call_claude_v3(
        bedrock_client,
        *,
        model_name: str,
        prompt: str,
        max_tokens_to_sample: int,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        assert any_are_none(top_k, top_p), "At least one of top_k, top_p must be None"
        bedrock_params = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens_to_sample,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }
        if system is not None:
            assert isinstance(system, str) and len(system) > 0
            bedrock_params["system"] = system

        if top_p is not None and temperature is not None:
            raise ValueError("Cannot specify both top_p and temperature; at most one must be specified.")

        if top_k is not None:
            assert isinstance(top_k, int) and len(system) >= 1
            bedrock_params["top_k"] = top_k
        elif top_p is not None:
            assert isinstance(top_p, (float, int)) and 0 <= top_p <= 1
            bedrock_params["top_p"] = top_p
        elif temperature is not None:
            assert isinstance(temperature, (float, int)) and 0 <= temperature <= 1
            bedrock_params["temperature"] = temperature

        if stop_sequences is not None:
            bedrock_params["stop_sequences"] = stop_sequences

        bedrock_params_json: str = json.dumps(bedrock_params)
        # print(f'\n\nbedrock_params_json:\n{json.dumps(bedrock_params, indent=4)}')
        response = bedrock_client.invoke_model(
            body=bedrock_params_json,
            modelId=model_name,
            accept="application/json",
            contentType="application/json",
        )
        response_body: Dict = json.loads(response.get("body").read())
        return "\n".join([d["text"] for d in response_body.get("content")])

    def call_claude_v3_messages_api(
        bedrock_client,
        *,
        model_name: str,
        prompt: str,
        max_tokens_to_sample: int,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Alternative implementation for calling Claude 3 models using the messages API.
        This version has simplified parameter handling with direct error propagation.

        Example usage:
            >>> bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
            >>> response = call_claude_v3_messages_api(
                    bedrock_client=bedrock_client,
                    model_name="anthropic.claude-3-sonnet-20240229-v1:0",
                    prompt="Tell me a joke",
                    max_tokens_to_sample=100
                )
        """
        messages = [{"role": "user", "content": prompt}]

        bedrock_params: Dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens_to_sample,
            "messages": messages,
        }

        ## Add optional parameters if provided:
        if system is not None:
            bedrock_params["system"] = system

        if temperature is not None:
            bedrock_params["temperature"] = temperature

        if top_p is not None:
            bedrock_params["top_p"] = top_p

        if top_k is not None:
            bedrock_params["top_k"] = top_k

        if stop_sequences is not None:
            bedrock_params["stop_sequences"] = stop_sequences

        response = bedrock_client.invoke_model(
            body=json.dumps(bedrock_params),
            modelId=model_name,
            accept="application/json",
            contentType="application/json",
        )

        response_body: Dict = json.loads(response.get("body").read())
        return "\n".join([d["text"] for d in response_body.get("content")])

    def call_bedrock(
        *,
        bedrock_client: Any,
        prompt: str,
        model_name: str,
        generation_params: Dict,
    ) -> str:
        """
        Call AWS Bedrock service to generate text from a prompt.

        Args:
            prompt (str): The input prompt for text generation
            model_name (str): The model ID to use for generation
            generation_params (Dict): Parameters for text generation
            bedrock_client (Any): Boto3 bedrock-runtime client

        Returns:
            str: The generated text

        Example usage:
            >>> bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
            >>> generated_text = call_bedrock(
                    prompt="Tell me a joke",
                    model_name="anthropic.claude-3-sonnet-20240229-v1:0",
                    generation_params={"max_tokens_to_sample": 100},
                    bedrock_client=bedrock_client
                )
        """
        if "anthropic.claude-3" in model_name:
            if "anthropic.claude-3-5" in model_name:
                ## Use the messages API implementation:
                generated_text: str = call_claude_v3_messages_api(
                    bedrock_client=bedrock_client,
                    prompt=prompt,
                    model_name=model_name,
                    **generation_params,
                )
            else:
                ## Use the original v3 implementation:
                generated_text: str = call_claude_v3(
                    bedrock_client=bedrock_client,
                    prompt=prompt,
                    model_name=model_name,
                    **generation_params,
                )
        elif "claude" in model_name:
            generated_text: str = call_claude_v1_v2(
                bedrock_client=bedrock_client,
                prompt=prompt,
                model_name=model_name,
                **generation_params,
            )
        else:
            bedrock_invoke_model_params = {"prompt": prompt, **generation_params}
            response = bedrock_client.invoke_model(
                body=json.dumps(bedrock_invoke_model_params),
                modelId=model_name,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())
            generated_text: str = response_body.get("completion")
        return generated_text

    class BedrockAccountConfig(Parameters):
        """
        Configuration for an AWS account to be used with Bedrock.

        Attributes:
            region_name (str): AWS region for this account
            role_arn (Optional[str]): IAM role ARN(s) for cross-account access
            session_timeout (int): Timeout in seconds for the assumed role session

        Example usage:
            >>> config = BedrockAccountConfig(
                    account_id="123456789012",
                    region_name="us-east-1",
                    role_arn="arn:aws:iam::123456789012:role/BedrockAccessRole"
                )
        """

        region_name: str
        role_arn: Optional[List[constr(min_length=1)]] = None
        session_timeout: conint(ge=1, le=43200) = 3600
        rpm: Optional[confloat(gt=0.0)] = None

        @model_validator(mode="before")
        @classmethod
        def set_bedrock_account_config_params(cls, params: Dict) -> Dict:
            Alias.set_region_name(params)
            Alias.set_role_arn(params)
            if params.get("role_arn") is not None:
                params["role_arn"] = as_list(params["role_arn"])
            return params

    class BedrockPrompter(GenerativeLM):
        aliases = ["bedrock"]
        executor: Optional[Any] = None
        bedrock_client: Optional[Any] = None
        boto3_session: Optional[Any] = None
        current_account_config: Optional[BedrockAccountConfig] = None

        class Hyperparameters(GenerativeLM.Hyperparameters):
            ALLOWED_TEXT_GENERATION_PARAMS: ClassVar[List[str]] = [
                "strategy",
                "name",
                "temperature",
                "top_k",
                "top_p",
                "max_new_tokens",
                "stop_sequences",
                "system",
            ]
            batch_size: Optional[conint(ge=1)] = None
            account_config: List[BedrockAccountConfig]
            model_name: constr(min_length=1)
            retries: conint(ge=0) = 2  ## Try 3 times
            retry_wait: confloat(ge=0) = 5.0
            retry_jitter: confloat(ge=0) = 0.5
            max_workers: Optional[conint(ge=1)] = None
            generation_params: Union[TextGenerationParams, Dict, str]
            config_selection_strategy: ConfigSelectionStrategy = ConfigSelectionStrategy.RANDOM_PER_REQUEST
            raise_on_error: bool = False

            @model_validator(mode="before")
            @classmethod
            def set_bedrock_params(cls, params: Dict) -> Dict:
                set_param_from_alias(
                    params,
                    param="account_config",
                    alias=["account_configs", "aws_account", "aws_accounts"],
                )
                set_param_from_alias(
                    params,
                    param="model_name",
                    alias=["model_id", "modelId", "model"],
                )
                set_param_from_alias(
                    params,
                    param="generation_params",
                    alias=[
                        "text_generation_params",
                        "generation",
                        "text_generation",
                        "generation_strategy",
                        "text_generation_strategy",
                    ],
                )
                Alias.set_max_workers(params)

                gen_params: Dict = params["generation_params"]
                extra_gen_params: Set[str] = set(gen_params.keys()) - set(cls.ALLOWED_TEXT_GENERATION_PARAMS)
                if len(extra_gen_params) != 0:
                    raise ValueError(
                        f"Following extra parameters for text generation are not allowed: {list(extra_gen_params)}; "
                        f"allowed parameters: {cls.ALLOWED_TEXT_GENERATION_PARAMS}."
                    )
                params["generation_params"] = TextGenerationParamsMapper.of(
                    params["generation_params"]
                ).initialize()

                params["account_config"] = as_list(params["account_config"])

                if params.get("max_workers") is not None:
                    assert isinstance(params["max_workers"], int) and params["max_workers"] >= 1
                    if params.get("batch_size") is None:
                        params["batch_size"] = params["max_workers"]
                return params

        def _select_account_config(self, model_idx: Tuple[int, int]) -> BedrockAccountConfig:
            """
            Select an account configuration based on the configured selection strategy.

            Returns:
                BedrockAccountConfig: The selected account configuration
            """
            if self.hyperparams.config_selection_strategy in {
                ConfigSelectionStrategy.RANDOM_AT_INIT,
                ConfigSelectionStrategy.RANDOM_PER_REQUEST,
            }:
                return random.choice(self.hyperparams.account_config)
            elif self.hyperparams.config_selection_strategy is ConfigSelectionStrategy.ROUND_ROBIN_AT_INIT:
                cur_model_idx, total_models = model_idx
                return self.hyperparams.account_config[cur_model_idx % len(self.hyperparams.account_config)]
            else:
                raise NotImplementedError(
                    f"Unsupported config selection strategy: {self.hyperparams.config_selection_strategy}"
                )

        @classmethod
        def _create_boto3_session(cls, account_config: BedrockAccountConfig) -> Any:
            """
            Create a boto3 session based on the provided account configuration.
            Uses IAMUtil for role assumption and session creation.

            Args:
                account_config (Optional[BedrockAccountConfig]): The account config to use for session creation

            Returns:
                Any: Boto3 session object
            """

            ## Use IAMUtil to create session with assumed role:
            return IAMUtil.create_session(
                role_arn=account_config.role_arn,
                region_name=account_config.region_name,
                default_max_duration=account_config.session_timeout,
                try_set_max_duration=False,
            )

        @property
        def max_num_generated_tokens(self) -> int:
            return self.hyperparams.generation_params.max_new_tokens

        def initialize(self, model_dir: Optional[FileMetadata] = None):
            ## Ignore the model_dir.
            try:
                self.refresh_session()

                ## Initialize executor:
                executor_params = {
                    "parallelize": Parallelize.sync,
                }
                if self.hyperparams.max_workers is not None:
                    executor_params["parallelize"] = Parallelize.threads
                    executor_params["max_workers"] = self.hyperparams.max_workers
                    if self.current_account_config.rpm is not None:
                        executor_params["max_calls_per_second"] = self.current_account_config.rpm / 60
                self.executor = dispatch_executor(**executor_params)

                cfg: BedrockAccountConfig = self.current_account_config
                if cfg.role_arn is not None:
                    cfg_msg: str = f" using IAM role {cfg.role_arn[-1]} in {cfg.region_name}."
                else:
                    cfg_msg: str = f" in {cfg.region_name}."
                Log.debug(f"Initialized BedrockPrompter {cfg_msg}")

            except Exception as e:
                Log.error(f"Failed to initialize BedrockPrompter:\n{String.format_exception_msg(e)}")
                raise e

        def refresh_session(self):
            ## Select an account configuration:
            self.current_account_config: BedrockAccountConfig = self._select_account_config(self.model_idx)

            ## Create boto3 session:
            self.boto3_session = self._create_boto3_session(self.current_account_config)

            ## Create bedrock client:
            self.bedrock_client = self.boto3_session.client(
                service_name="bedrock-runtime",
                region_name=self.current_account_config.region_name,
            )

        def cleanup(self):
            super(self.__class__, self).cleanup()
            stop_executor(self.executor)
            self.bedrock_client = None
            self.boto3_session = None

        @property
        def bedrock_text_generation_params(self) -> Dict[str, Any]:
            ## https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html
            generation_params: TextGenerationParams = self.hyperparams.generation_params
            bedrock_params: Dict[str, Any] = {
                "max_tokens_to_sample": generation_params.max_new_tokens,
            }
            for param in remove_values(
                self.hyperparams.ALLOWED_TEXT_GENERATION_PARAMS,
                ["strategy", "name", "max_new_tokens"],
            ):
                if hasattr(generation_params, param) and getattr(generation_params, param) is not None:
                    bedrock_params[param] = getattr(generation_params, param)
            return bedrock_params

        def bedrock_error_handler(self, e: Exception) -> bool:
            if isinstance(e, ClientError):
                if e.response["Error"]["Code"] == "ExpiredToken":
                    self.refresh_session()
                    return True
                if e.response["Error"]["Code"] == "ThrottlingException":
                    return True
                if e.response["Error"]["Code"] == "AccessDeniedException":
                    Log.error(
                        f"Access denied for model {self.hyperparams.model_name} when using account config:\n"
                        f"{self.current_account_config}"
                    )
                    return False  ## Not recoverable

        def prompt_model_with_retries(self, prompt: str) -> str:
            if self.hyperparams.config_selection_strategy is ConfigSelectionStrategy.RANDOM_PER_REQUEST:
                self.refresh_session()

            if self.bedrock_client is None:
                raise SystemError("BedrockPrompter not initialized. Call initialize() first.")

            try:
                return retry(
                    call_bedrock,
                    bedrock_client=self.bedrock_client,
                    prompt=prompt,
                    model_name=self.hyperparams.model_name,
                    generation_params=self.bedrock_text_generation_params,
                    retries=self.hyperparams.retries,
                    wait=self.hyperparams.retry_wait,
                    jitter=self.hyperparams.retry_jitter,
                    error_handler=self.bedrock_error_handler,
                    silent=True,
                )
            except Exception as e:
                if self.hyperparams.raise_on_error:
                    raise e
                Log.error(String.format_exception_msg(e))
                return ""

        def predict_step(self, batch: Prompts, **kwargs) -> Any:
            generated_texts: List = []
            for prompt in batch.prompts().tolist():
                ## Template has already been applied
                generated_text: Any = dispatch(
                    self.prompt_model_with_retries,
                    prompt,
                    executor=self.executor,
                    parallelize=Parallelize.sync
                    if self.hyperparams.max_workers is None
                    else Parallelize.threads,
                )
                generated_texts.append(generated_text)
            generated_texts: List[str] = accumulate(generated_texts)
            return {GENERATED_TEXTS_COL: generated_texts}
