# app/secrets.py
import os

import boto3


def load_secrets_from_ssm() -> None:
    """On Lambda, fetch API keys from SSM Parameter Store at cold start.

    For each variable named *_PARAM (e.g. GROQ_API_KEY_PARAM=/platform/groq-api-key),
    read the SecureString parameter and expose it as the variable without the suffix
    (GROQ_API_KEY). Locally there are no *_PARAM variables, so .env is used instead.
    """
    wanted = {k[: -len("_PARAM")]: v for k, v in os.environ.items() if k.endswith("_PARAM")}
    if not wanted:
        return
    ssm = boto3.client("ssm")
    response = ssm.get_parameters(Names=list(wanted.values()), WithDecryption=True)
    by_name = {p["Name"]: p["Value"] for p in response["Parameters"]}
    for env_name, param_name in wanted.items():
        if param_name in by_name:
            os.environ[env_name] = by_name[param_name]
