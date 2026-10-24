from att_agent.education.relevance_verifier import evidence_matches, verified_search


def test_whitespace_normalization():
    assert evidence_matches("family would grow and bless", "family would grow\nand bless")


def test_no_fabricated_evidence():
    assert not evidence_matches("family was 1000 people", "family would grow")


def test_verified_search_filters_rejected(monkeypatch):
    from att_agent.education import relevance_verifier as verifier

    monkeypatch.setattr(
        verifier,
        "retrieve",
        lambda *args, **kwargs: [
            {"search_id": "1", "subject": "bible", "title": "A"}
        ],
    )
    monkeypatch.setattr(verifier, "verify_one", lambda *args, **kwargs: (None, "irrelevant"))
    result = verified_search(object(), "a question", subject="bible")
    assert result["results"] == [] and result["rejected_count"] == 1
