import json
from types import SimpleNamespace

from att_agent.education.relevance_verifier import (
    evidence_matches,
    verified_search,
    verify_one,
)


class FakeServingEndpoints:
    def __init__(self, decision):
        self.decision = decision

    def query(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.decision)))]
        )


def fake_workspace(decision):
    return SimpleNamespace(serving_endpoints=FakeServingEndpoints(decision))


MATH_DOC = {
    "search_id": "math-1",
    "subject": "mathematics",
    "content_type": "EXERCISE",
    "title": "Geometry Question 03",
    "searchable_text": "A right triangle has legs 6 cm and 8 cm. Find x.",
}

BIBLE_DOC = {
    "search_id": "bible-1",
    "subject": "bible",
    "content_type": "KNOWLEDGE",
    "title": "Abraham and Sarah's Promise",
    "searchable_text": (
        "The Lord promised the prophet Abraham and his wife Sarah that "
        "their family would grow\nand bless the whole world."
    ),
}


def test_whitespace_normalization():
    assert evidence_matches("family would grow and bless", "family would grow\nand bless")


def test_no_fabricated_evidence():
    assert not evidence_matches("family was 1000 people", "family would grow")


def test_math_exercise_accepted_as_resource():
    ws = fake_workspace(
        {
            "relevant": True,
            "evidence": "A right triangle has legs 6 cm and 8 cm. Find x.",
            "reason": "Matching exercise without a worked explanation",
            "match_type": "EXERCISE_MATCH",
            "answer_supported": False,
        }
    )
    result, reason = verify_one(
        ws, "How do I calculate the hypotenuse?", MATH_DOC, "model", "auto"
    )
    assert reason == "accepted"
    assert result["match_type"] == "EXERCISE_MATCH"
    assert result["answer_supported"] is False


def test_answer_question_rejects_solution_free_exercise():
    ws = fake_workspace(
        {
            "relevant": True,
            "evidence": "A right triangle has legs 6 cm and 8 cm. Find x.",
            "reason": "A related exercise, not an answer",
            "match_type": "EXERCISE_MATCH",
            "answer_supported": False,
        }
    )
    result, reason = verify_one(
        ws, "How do I calculate the hypotenuse?", MATH_DOC, "model", "answer_question"
    )
    assert result is None
    assert "answer" in reason


def test_bible_answer_remains_supported():
    ws = fake_workspace(
        {
            "relevant": True,
            "evidence": (
                "The Lord promised the prophet Abraham and his wife Sarah "
                "that their family would grow and bless the whole world."
            ),
            "reason": "Direct passage evidence",
            "match_type": "ANSWER_SUPPORT",
            "answer_supported": True,
        }
    )
    result, reason = verify_one(
        ws, "What promise did Abraham receive?", BIBLE_DOC, "model", "answer_question"
    )
    assert reason == "accepted"
    assert result["answer_supported"] is True


def test_fabricated_evidence_still_rejected():
    ws = fake_workspace(
        {
            "relevant": True,
            "evidence": "The Lord gave Abraham 1000 sheep.",
            "reason": "Claimed relevance",
            "match_type": "ANSWER_SUPPORT",
            "answer_supported": True,
        }
    )
    result, reason = verify_one(ws, "What did Abraham receive?", BIBLE_DOC, "model")
    assert result is None
    assert "evidence not found" in reason


def test_verified_search_filters_rejected(monkeypatch):
    from att_agent.education import relevance_verifier as verifier

    monkeypatch.setattr(
        verifier,
        "retrieve",
        lambda *args, **kwargs: [{"search_id": "1", "subject": "bible", "title": "A"}],
    )
    monkeypatch.setattr(verifier, "verify_one", lambda *args, **kwargs: (None, "irrelevant"))
    result = verified_search(object(), "a question", subject="bible")
    assert result["results"] == [] and result["rejected_count"] == 1
    assert result["search_intent"] == "auto"
