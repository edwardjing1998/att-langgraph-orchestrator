"""Databricks client dedicated to the Education RAG workspace."""

import os

from databricks.sdk import WorkspaceClient


def get_education_workspace_client() -> WorkspaceClient:
    """Use Education-specific credentials without altering legacy settings."""
    host = os.environ["EDUCATION_DATABRICKS_HOST"].strip().rstrip("/")
    client_id = os.getenv("EDUCATION_DATABRICKS_CLIENT_ID", "").strip()
    client_secret = os.getenv("EDUCATION_DATABRICKS_CLIENT_SECRET", "").strip()
    token = os.getenv("EDUCATION_DATABRICKS_TOKEN", "").strip()

    if client_id and client_secret:
        return WorkspaceClient(
            host=host,
            client_id=client_id,
            client_secret=client_secret,
            auth_type="oauth-m2m",
        )
    if token:
        return WorkspaceClient(host=host, token=token, auth_type="pat")
    raise RuntimeError(
        "Education Databricks credentials are not configured. "
        "Set EDUCATION_DATABRICKS_CLIENT_ID and "
        "EDUCATION_DATABRICKS_CLIENT_SECRET, or "
        "EDUCATION_DATABRICKS_TOKEN."
    )
