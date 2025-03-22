import logging
from typing import Dict, List, Optional
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from fmcore.types.enums.metric_enums import EvaluationFieldType

logger = logging.getLogger(__name__)


class DeepEvalUtils:
    """
    Utility class for working with DeepEval framework.

    This class provides helper methods for mapping between our framework's
    field types and DeepEval's expected structure.
    """

    # Default mapping between our framework field types and DeepEval field types
    DEFAULT_FIELD_MAPPING = {
        EvaluationFieldType.INPUT: LLMTestCaseParams.INPUT,
        EvaluationFieldType.OUTPUT: LLMTestCaseParams.ACTUAL_OUTPUT,
        EvaluationFieldType.GROUND_TRUTH: LLMTestCaseParams.EXPECTED_OUTPUT,
        EvaluationFieldType.CONTEXT: LLMTestCaseParams.CONTEXT,
    }

    # Parameters that should be treated as lists in LLMTestCase
    LIST_PARAMS = {"context", "retrieval_context", "expected_tools", "tools_called"}

    @classmethod
    def map_data_to_testcase(
        cls, data: Dict, field_mapping: Dict[EvaluationFieldType, str] = None
    ) -> LLMTestCase:
        """
        Convert input data into a DeepEval LLMTestCase using a specified field mapping.

        This method takes a dictionary of input data and maps it to the expected structure
        of a DeepEval LLMTestCase. The mapping is guided by a field mapping dictionary that
        specifies how our framework's field types correspond to the dataset columns.

        Args:
            data (Dict): A dictionary containing the input data fields from the customer dataset.
            field_mapping (Optional[Dict[str, str]]): A dictionary that explicitly maps our framework's
                field types to the corresponding dataset columns. If not provided, default mappings are used.

        Returns:
            LLMTestCase: An instance of LLMTestCase populated with the mapped data.
        """
        testcase_params = {}

        if not field_mapping:
            # If no mapping provided, return params for all default fields
            field_mapping = cls.DEFAULT_FIELD_MAPPING

        for field_type_str, customer_field in field_mapping.items():
            # Convert string to our enum (e.g. "INPUT" -> EvaluationFieldType.INPUT)
            evaluation_field_type = EvaluationFieldType.from_str(field_type_str)

            # Look up corresponding DeepEval param from our default mapping
            deepeval_field_type = cls.DEFAULT_FIELD_MAPPING.get(evaluation_field_type)
            if deepeval_field_type:
                # Extract the value from the input data using the customer field name
                value = data[customer_field]
                # If the field type is CONTEXT, ensure the value is a list
                if deepeval_field_type == LLMTestCaseParams.CONTEXT:
                    value = [value]
                # Add the mapped value to the testcase parameters
                testcase_params[deepeval_field_type.value] = value

        return LLMTestCase(**testcase_params)

    @classmethod
    def infer_evaluation_params(
        cls, field_mapping: Optional[Dict[str, str]] = None
    ) -> List[LLMTestCaseParams]:
        """
        Infers the evaluation parameters for DeepEval by using the provided field mapping.
        If no custom field mapping is provided, defaults from DeepEval are used.

        Args:
            field_mapping: Mapping from field type strings to dataset columns (e.g. {"INPUT": "question"})

        Returns:
            List of LLMTestCaseParams corresponding to the field types in the mapping

        Example:
            >>> field_mapping = {"INPUT": "question", "RESPONSE": "actual_output"}
            >>> params = DeepEvalUtils.infer_evaluation_params(field_mapping)
            >>> # Returns [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
        """
        if not field_mapping:
            # If no mapping provided, return params for all default fields
            return list(cls.DEFAULT_FIELD_MAPPING.values())

        inferred_params = []
        for evaluation_field_type_str, customer_field_str in field_mapping.items():
            try:
                # Convert string to our enum (e.g. "INPUT" -> EvaluationFieldType.INPUT)
                evaluation_field_type = EvaluationFieldType.from_str(evaluation_field_type_str)

                # Look up corresponding DeepEval param from our default mapping
                if deepeval_field_type := cls.DEFAULT_FIELD_MAPPING.get(evaluation_field_type):
                    if deepeval_field_type not in inferred_params:
                        inferred_params.append(deepeval_field_type)
            except KeyError:
                logger.warning(f"Unrecognized field type in mapping: {field_type_str}")
                continue

        return inferred_params
