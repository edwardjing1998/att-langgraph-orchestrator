import json
from types import SimpleNamespace

from att_agent.education.question_understanding import understand_question


def client(response):
    def query(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])

    return SimpleNamespace(serving_endpoints=SimpleNamespace(query=query))


def test_corrects_typo_and_preserves_subject():
    ws = client(
        json.dumps(
            {
                "normalized_query": "Grade 3 mathematics knowledge about triangles",
                "subject": "science",
                "content_type": "KNOWLEDGE",
                "grade": 3,
                "topic": "triangles",
            }
        )
    )
    result = understand_question(ws, "Grade 3 knowledges about triangles", "mathematics", "model")
    assert result.subject == "mathematics"
    assert result.content_type == "KNOWLEDGE"
    assert result.grade == 3
    assert "knowledge" in result.normalized_query


def test_bad_json_preserves_original_question():
    result = understand_question(client("not json"), "exercisess on areas", "mathematics", "model")
    assert result.normalized_query == "exercisess on areas"
    assert result.content_type == "ANY"


def test_rejects_invented_grade():
    ws = client(
        json.dumps(
            {
                "normalized_query": "triangles",
                "subject": "mathematics",
                "content_type": "ANY",
                "grade": 500,
                "topic": "triangles",
            }
        )
    )
    assert understand_question(ws, "triangles", "", "model").normalized_query == "triangles"
