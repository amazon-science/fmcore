import inspect
import enum
import json
import textwrap
from typing import Dict, Literal

from dspy.adapters.chat_adapter import (
    enumerate_fields,
    format_fields,
    get_dspy_field_type,
    prepare_schema,
)
from dspy.adapters.chat_adapter import FieldInfoWithName, BuiltInCompletedOutputFieldInfo, FieldInfo
from dspy.signatures.signature import SignatureMeta


def custom_prepare_instructions(signature: SignatureMeta):
    """
    Prepares structured instructions for the DSPy prompt optimizer with the following order:

    +---------------------+
    |     Objective       |
    +---------------------+
    |   Input Fields      |
    +---------------------+
    |   Output Fields     |
    +---------------------+
    |     Examples        |
    +---------------------+

    This custom structure differs from the default DSPy structure:
    Default DSPy Structure:
    +---------------------+
    |     Inputs          |
    +---------------------+
    |     Outputs         |
    +---------------------+
    |     Examples        |
    +---------------------+
    |     Objective       |
    +---------------------+

    The custom order improves accuracy by placing the objective before the fields.

    Args:
        signature (SignatureMeta): Contains metadata about the prompt’s input/output fields and instructions.

    Returns:
        str: The formatted prompt instructions with the custom structure.

    Example:
        signature = SignatureMeta( ... )
        instructions = custom_prepare_instructions(signature)


    For more information, refer to the DSPy documentation:
        https://github.com/stanfordnlp/dspy/blob/main/dspy/adapters/chat_adapter.py
    """

    parts = []

    instructions = textwrap.dedent(signature.instructions)
    objective = "\n".join(instructions.splitlines())
    parts.append(objective)

    parts.append("Your input fields are:\n" + enumerate_fields(signature.input_fields))
    parts.append("Your output fields are:\n" + enumerate_fields(signature.output_fields))
    parts.append(
        "All interactions will be structured in the following way, with the appropriate values filled in."
    )

    def field_metadata(field_name, field_info):
        field_type = field_info.annotation

        if get_dspy_field_type(field_info) == "input" or field_type is str:
            desc = ""
        elif field_type is bool:
            desc = "must be True or False"
        elif field_type in (int, float):
            desc = f"must be a single {field_type.__name__} value"
        elif inspect.isclass(field_type) and issubclass(field_type, enum.Enum):
            desc = f"must be one of: {'; '.join(field_type.__members__)}"
        elif hasattr(field_type, "__origin__") and field_type.__origin__ is Literal:
            desc = (
                # Strongly encourage the LM to avoid choosing values that don't appear in the
                # literal or returning a value of the form 'Literal[<selected_value>]'
                f"must exactly match (no extra characters) one of: {'; '.join([str(x) for x in field_type.__args__])}"
            )
        else:
            desc = "must adhere to the JSON schema: "
            desc += json.dumps(prepare_schema(field_type), ensure_ascii=False)

        desc = (" " * 8) + f"# note: the value you produce {desc}" if desc else ""
        return f"{{{field_name}}}{desc}"

    def format_signature_fields_for_instructions(fields: Dict[str, FieldInfo]):
        return format_fields(
            fields_with_values={
                FieldInfoWithName(name=field_name, info=field_info): field_metadata(field_name, field_info)
                for field_name, field_info in fields.items()
            },
        )

    parts.append(format_signature_fields_for_instructions(signature.input_fields))
    parts.append(format_signature_fields_for_instructions(signature.output_fields))
    parts.append(format_fields({BuiltInCompletedOutputFieldInfo: ""}))

    return "\n\n".join(parts).strip()
