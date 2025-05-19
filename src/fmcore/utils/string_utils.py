def is_empty_or_none(role_arn: str) -> bool:
    return not role_arn or not role_arn.strip()
