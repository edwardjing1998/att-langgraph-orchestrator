from att_agent.graph import _extractive_fallback
from att_agent.models import Evidence


def test_extractive_fallback_contains_document_citation() -> None:
    evidence = [
        Evidence(
            source_platform="databricks",
            source_type="policy",
            document_id="POL-1001",
            title="Roaming adjustment policy",
            content="An incorrect roaming charge may be reviewed.",
            source_reference="policy/POL-1001",
            access_level="INTERNAL",
        )
    ]

    answer = _extractive_fallback(evidence)

    assert "[POL-1001]" in answer
    assert "incorrect roaming charge" in answer
