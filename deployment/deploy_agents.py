"""Deploy the Document Analyst via the `databricks-agents` SDK (Bonus B)."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow

_ROOT = Path(__file__).resolve().parent.parent
_AGENT_MODEL_PATH = "deployment/agent_model.py"

_PIP_REQUIREMENTS = [
    "mlflow", "langgraph", "langchain-core", "langchain-openai",
    "databricks-langchain", "databricks-vectorsearch", "databricks-sdk",
    "databricks-agents", "langchain-mcp-adapters", "mcp", "openai", 
    "pydantic", "python-dotenv",
]


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise OSError(f"Missing required environment variable: {name}")
    return val


def deploy_with_agents_sdk():
    from databricks import agents
    from mlflow.models.resources import DatabricksServingEndpoint

    uc_catalog = _require("UC_CATALOG")
    uc_schema = _require("UC_SCHEMA")
    model_name = _require("SERVING_ENDPOINT_NAME").replace("-", "_") + "_v2"
    uc_name = f"{uc_catalog}.{uc_schema}.{model_name}"
    llm_endpoint = _require("DATABRICKS_MODEL")

    mlflow.set_registry_uri("databricks-uc")

    # Agent Framework deployment requires pyfunc.log_model with resources
    # (not langchain.log_model) to generate the ChatCompletion-compatible schema
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=_AGENT_MODEL_PATH,
            artifact_path="agent",
            code_paths=[
                str(_ROOT / "agent"), str(_ROOT / "rag"),
                str(_ROOT / "tools"), str(_ROOT / "config.py"),
            ],
            pip_requirements=_PIP_REQUIREMENTS,
            resources=[
                # Declares dependency on the LLM endpoint for automatic auth
                DatabricksServingEndpoint(endpoint_name=llm_endpoint),
            ],
            input_example={"messages": [{"role": "user", "content": "What was the revenue in 2023?"}]},
        )

    registered = mlflow.register_model(model_info.model_uri, uc_name)
    print(f"Registered model: {uc_name}, version {registered.version}")

    deployment = agents.deploy(
        model_name=uc_name,
        model_version=registered.version,
        scale_to_zero=True,
    )
    print(f"Endpoint name:   {deployment.endpoint_name}")
    print(f"Review App URL:  {deployment.review_app_url}")
    return deployment


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    deploy_with_agents_sdk()
