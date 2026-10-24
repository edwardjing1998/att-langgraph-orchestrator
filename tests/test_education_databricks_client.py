"""Ensure education credentials cannot be confused with legacy credentials."""

from att_agent.education import databricks_client


def test_education_uses_its_own_oauth_settings(monkeypatch):
    calls = []
    monkeypatch.setattr(databricks_client, "WorkspaceClient", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setenv("DATABRICKS_HOST", "https://legacy.example")
    monkeypatch.setenv("DATABRICKS_TOKEN", "legacy")
    monkeypatch.setenv("EDUCATION_DATABRICKS_HOST", "https://education.example/")
    monkeypatch.setenv("EDUCATION_DATABRICKS_CLIENT_ID", "education-id")
    monkeypatch.setenv("EDUCATION_DATABRICKS_CLIENT_SECRET", "education-secret")
    monkeypatch.delenv("EDUCATION_DATABRICKS_TOKEN", raising=False)

    databricks_client.get_education_workspace_client()
    assert calls == [
        {
            "host": "https://education.example",
            "client_id": "education-id",
            "client_secret": "education-secret",
            "auth_type": "oauth-m2m",
        }
    ]


def test_education_does_not_fallback_to_legacy_token(monkeypatch):
    monkeypatch.setenv("DATABRICKS_TOKEN", "legacy")
    monkeypatch.setenv("EDUCATION_DATABRICKS_HOST", "https://education.example")
    for name in (
        "EDUCATION_DATABRICKS_CLIENT_ID",
        "EDUCATION_DATABRICKS_CLIENT_SECRET",
        "EDUCATION_DATABRICKS_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)

    import pytest

    with pytest.raises(
        RuntimeError,
        match="Education Databricks credentials are not configured",
    ):  # no legacy fallback
        databricks_client.get_education_workspace_client()
