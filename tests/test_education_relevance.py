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
    result, reason = verify_one(ws, "How do I calculate the hypotenuse?", MATH_DOC, "model", "auto")
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


def test_retrieve_filters_inside_ann_query():
    from att_agent.education.relevance_verifier import retrieve

    class API:
        def __init__(self):
            self.body = None

        def do(self, **kwargs):
            self.body = kwargs["body"]
            return {"manifest": {"columns": []}, "result": {"data_array": []}}

    api = API()
    retrieve(
        SimpleNamespace(api_client=api),
        "triangle knowledge",
        8,
        subject="mathematics",
        content_type="KNOWLEDGE",
    )
    assert api.body["query_type"] == "ANN"
    assert json.loads(api.body["filters_json"]) == {
        "subject": "mathematics",
        "content_type": "KNOWLEDGE",
    }


def test_knowledge_rejects_exercise_without_calling_model():
    ws = SimpleNamespace(serving_endpoints=None)
    result, reason = verify_one(
        ws, "triangle knowledges", MATH_DOC, "model", requested_content_type="KNOWLEDGE"
    )
    assert result is None
    assert "content type" in reason


def test_general_pages_query_does_not_filter_to_knowledge(monkeypatch):
    from att_agent.education import relevance_verifier as verifier
    from att_agent.education.question_understanding import UnderstoodQuestion

    seen = []
    rectangle = {
        "search_id": "rect-1",
        "title": "Rectangle Card Sort Activity",
        "subject": "mathematics",
        "content_type": "EXERCISE",
        "searchable_text": "Sort the rectangle cards into categories.",
    }

    monkeypatch.setattr(
        verifier,
        "understand_question",
        lambda *args: UnderstoodQuestion(
            "Find mathematics knowledge about rectangles", "mathematics", "KNOWLEDGE"
        ),
    )

    def retrieve_stub(*args, **kwargs):
        seen.append(kwargs["content_type"])
        return [rectangle]

    monkeypatch.setattr(verifier, "retrieve", retrieve_stub)

    def verify_stub(*args, **kwargs):
        seen.append(kwargs["requested_content_type"])
        return ({"search_id": "rect-1"}, "accepted")

    monkeypatch.setattr(verifier, "verify_one", verify_stub)
    result = verified_search(
        object(), "Can you show me pages related to rectangle?", subject="mathematics"
    )
    assert [item["search_id"] for item in result["results"]] == ["rect-1"]
    assert seen == ["ANY", "ANY"]


def test_explicit_exercises_stay_filtered(monkeypatch):
    from att_agent.education import relevance_verifier as verifier

    monkeypatch.setenv("EDUCATION_RELATED_RESULTS_ENABLED", "false")
    seen = []

    def retrieve_stub(*args, **kwargs):
        seen.append(kwargs["content_type"])
        return []

    monkeypatch.setenv("EDUCATION_QUERY_NORMALIZATION_ENABLED", "false")
    monkeypatch.setattr(verifier, "retrieve", retrieve_stub)
    result = verified_search(object(), "Find rectangle exercisess", subject="mathematics")
    assert result["results"] == []
    assert seen == ["EXERCISE"]


def test_original_query_retry_recovers_rectangle(monkeypatch):
    from att_agent.education import relevance_verifier as verifier
    from att_agent.education.question_understanding import UnderstoodQuestion

    monkeypatch.setattr(
        verifier,
        "understand_question",
        lambda *args: UnderstoodQuestion("Unhelpful rewrite", "mathematics", "ANY"),
    )
    called_queries = []

    def retrieve_stub(_ws, question, *args, **kwargs):
        called_queries.append(question)
        if question == "Unhelpful rewrite":
            return []
        return [{"search_id": "rect-1", "title": "Rectangle Card Sort Activity"}]

    monkeypatch.setattr(verifier, "retrieve", retrieve_stub)
    monkeypatch.setattr(
        verifier, "verify_one", lambda *args, **kwargs: ({"search_id": "rect-1"}, "accepted")
    )
    results = verified_search(object(), "pages related to rectangle", subject="mathematics")
    assert [item["search_id"] for item in results["results"]] == ["rect-1"]
    assert called_queries == ["Unhelpful rewrite", "pages related to rectangle"]


def test_category_browse_detection():
    from att_agent.education.relevance_verifier import is_category_browse

    assert is_category_browse("Can you show me all the pages related to knowledge?", "KNOWLEDGE")
    assert is_category_browse("Show me all exercises", "EXERCISE")
    assert is_category_browse("show me all the exercisess pages", "EXERCISE")
    assert not is_category_browse("show me knowledge about triangles", "KNOWLEDGE")
    assert not is_category_browse("show me rectangle exercises", "EXERCISE")
    assert not is_category_browse("show me rectangle pages", "ANY")


def test_knowledge_browse_skips_vector_search_and_model(monkeypatch):
    from att_agent.education import relevance_verifier as verifier

    monkeypatch.setenv("EDUCATION_APPROVED_SOURCE_TABLE", "education_rag.gold.approved_source")
    monkeypatch.setenv("EDUCATION_SQL_WAREHOUSE_ID", "test-warehouse")
    monkeypatch.setenv("EDUCATION_QUERY_NORMALIZATION_ENABLED", "false")
    monkeypatch.setattr(
        verifier,
        "retrieve",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ANN should not run")),
    )
    monkeypatch.setattr(
        verifier,
        "verify_one",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM should not run")),
    )
    monkeypatch.setattr(
        verifier,
        "browse_by_content_type",
        lambda ws, subject, content_type, limit: [
            {
                "search_id": "tri-1",
                "subject": "mathematics",
                "title": "Classifying Triangles by Sides",
                "content_type": "KNOWLEDGE",
                "searchable_text": "Triangles have three sides",
                "image_links_json": "[]",
            }
        ],
    )
    result = verifier.verified_search(
        object(),
        "Can you show me all the pages related to knowledge?",
        subject="mathematics",
        search_intent="find_resources",
    )
    assert [doc["search_id"] for doc in result["results"]] == ["tri-1"]
    assert result["results"][0]["content_type"] == "KNOWLEDGE"


def test_browse_sql_uses_metadata_and_parameters(monkeypatch):
    from att_agent.education.relevance_verifier import browse_by_content_type

    monkeypatch.setenv("EDUCATION_APPROVED_SOURCE_TABLE", "education_rag.gold.approved_source")
    monkeypatch.setenv("EDUCATION_SQL_WAREHOUSE_ID", "test-warehouse")

    class API:
        def __init__(self):
            self.body = None

        def do(self, **kwargs):
            self.body = kwargs["body"]
            return {
                "status": {"state": "SUCCEEDED"},
                "manifest": {
                    "schema": {"columns": [{"name": "search_id"}, {"name": "content_type"}]}
                },
                "result": {"data_array": [["tri-1", "KNOWLEDGE"]]},
            }

    api = API()
    result = browse_by_content_type(SimpleNamespace(api_client=api), "mathematics", "KNOWLEDGE", 5)
    assert result == [{"search_id": "tri-1", "content_type": "KNOWLEDGE"}]
    assert "content_type = :kind" in api.body["statement"]
    assert any(p["value"] == "KNOWLEDGE" for p in api.body["parameters"])


def test_related_results_when_strict_verification_rejects(monkeypatch):
    from att_agent.education import relevance_verifier as verifier

    monkeypatch.setenv("EDUCATION_QUERY_NORMALIZATION_ENABLED", "false")
    monkeypatch.setenv("EDUCATION_RELATED_RESULTS_ENABLED", "true")
    monkeypatch.setattr(verifier, "retrieve", lambda *args, **kwargs: [MATH_DOC])
    monkeypatch.setattr(verifier, "verify_one", lambda *args, **kwargs: (None, "no answer"))
    ws = fake_workspace({"related": True, "reason": "An exercise in the requested category"})
    response = verifier.verified_search(
        ws,
        "Show me lessons related to exercise?",
        subject="mathematics",
        search_intent="find_resources",
    )
    assert response["results"] == []
    assert response["match_status"] == "related_results"
    assert response["related_results"][0]["search_id"] == "math-1"
    assert response["related_results"][0]["content_type"] == "EXERCISE"
    assert response["related_results"][0]["answer_supported"] is False


def test_related_results_reject_wrong_subject_and_content_type():
    from att_agent.education.relevance_verifier import assess_related_resource

    ws = fake_workspace({"related": True, "reason": "maybe related"})
    assert (
        assess_related_resource(
            ws,
            "Find exercises",
            BIBLE_DOC,
            "endpoint",
            subject="mathematics",
            requested_content_type="EXERCISE",
        )
        is None
    )
    assert (
        assess_related_resource(
            ws,
            "Find exercises",
            MATH_DOC,
            "endpoint",
            subject="mathematics",
            requested_content_type="KNOWLEDGE",
        )
        is None
    )


def test_related_results_disabled(monkeypatch):
    from att_agent.education import relevance_verifier as verifier

    monkeypatch.setenv("EDUCATION_QUERY_NORMALIZATION_ENABLED", "false")
    monkeypatch.setenv("EDUCATION_RELATED_RESULTS_ENABLED", "false")
    monkeypatch.setattr(verifier, "retrieve", lambda *args, **kwargs: [MATH_DOC])
    monkeypatch.setattr(verifier, "verify_one", lambda *args, **kwargs: (None, "not accepted"))
    response = verifier.verified_search(object(), "Find triangle exercises", subject="mathematics")
    assert response["related_results"] == []
    assert response["match_status"] == "no_results"


def test_related_results_not_added_when_verified_results_exist(monkeypatch):
    from att_agent.education import relevance_verifier as verifier

    monkeypatch.setenv("EDUCATION_QUERY_NORMALIZATION_ENABLED", "false")
    monkeypatch.setattr(verifier, "retrieve", lambda *args, **kwargs: [MATH_DOC])
    monkeypatch.setattr(
        verifier,
        "verify_one",
        lambda *args, **kwargs: ({"search_id": "math-1", "content_type": "EXERCISE"}, "accepted"),
    )
    response = verifier.verified_search(object(), "Find triangle exercises", subject="mathematics")
    assert response["related_results"] == []
    assert response["match_status"] == "matches"


def test_related_suggestions_deduplicates_and_limits(monkeypatch):
    from att_agent.education import relevance_verifier as verifier

    monkeypatch.setenv("EDUCATION_RELATED_MAX_CHECKS", "2")
    checked = []

    def stub(_ws, question, doc, endpoint, subject="", requested_content_type="ANY"):
        checked.append(doc["search_id"])
        return {"search_id": doc["search_id"]}

    monkeypatch.setattr(verifier, "assess_related_resource", stub)
    documents = [dict(MATH_DOC, search_id=i) for i in ("a", "a", "b", "c")]
    result = verifier.related_suggestions(
        object(), "question", documents, "endpoint", "", "ANY", 10
    )
    assert [item["search_id"] for item in result] == ["a", "b"]
    assert checked == ["a", "b"]


def test_ai_browse_lessons_offer_exercise_records_without_mislabeling(monkeypatch):
    from att_agent.education import relevance_verifier as verifier
    from att_agent.education.question_understanding import UnderstoodQuestion

    monkeypatch.setenv("EDUCATION_QUERY_NORMALIZATION_ENABLED", "true")
    monkeypatch.setenv("EDUCATION_APPROVED_SOURCE_TABLE", "education_rag.gold.approved_source")
    monkeypatch.setenv("EDUCATION_SQL_WAREHOUSE_ID", "test-warehouse")
    monkeypatch.setattr(
        verifier,
        "understand_question",
        lambda *args: UnderstoodQuestion(
            normalized_query="exercise resources",
            subject="mathematics",
            content_type="EXERCISE",
            topic="",
            search_mode="browse",
            resource_format="lesson",
        ),
    )
    monkeypatch.setattr(
        verifier,
        "browse_by_content_type",
        lambda *_args: [MATH_DOC, dict(MATH_DOC, search_id="math-2")],
    )
    monkeypatch.setattr(
        verifier,
        "retrieve",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ANN should not run")),
    )
    response = verifier.verified_search(
        object(),
        "Can you show me lessons related to exercises?",
        subject="mathematics",
    )
    assert response["results"] == []
    assert response["match_status"] == "related_results"
    assert [item["search_id"] for item in response["related_results"]] == ["math-1", "math-2"]
    assert all(item["content_type"] == "EXERCISE" for item in response["related_results"])
    assert all(item["match_type"] == "RELATED_RESOURCE" for item in response["related_results"])


def test_ai_browse_pages_still_returns_direct_category_results(monkeypatch):
    from att_agent.education import relevance_verifier as verifier
    from att_agent.education.question_understanding import UnderstoodQuestion

    monkeypatch.setenv("EDUCATION_APPROVED_SOURCE_TABLE", "education_rag.gold.approved_source")
    monkeypatch.setenv("EDUCATION_SQL_WAREHOUSE_ID", "test-warehouse")
    monkeypatch.setattr(
        verifier,
        "understand_question",
        lambda *args: UnderstoodQuestion(
            "exercise pages",
            "mathematics",
            "EXERCISE",
            search_mode="browse",
            resource_format="page",
        ),
    )
    monkeypatch.setattr(verifier, "browse_by_content_type", lambda *_args: [MATH_DOC])
    response = verifier.verified_search(object(), "Show pages related to exercises", "mathematics")
    assert [item["search_id"] for item in response["results"]] == ["math-1"]
    assert response["related_results"] == []


def test_ai_browse_without_warehouse_uses_category_vector_search(monkeypatch):
    from att_agent.education import relevance_verifier as verifier
    from att_agent.education.question_understanding import UnderstoodQuestion

    monkeypatch.delenv("EDUCATION_APPROVED_SOURCE_TABLE", raising=False)
    monkeypatch.delenv("EDUCATION_SQL_WAREHOUSE_ID", raising=False)
    monkeypatch.setenv("EDUCATION_RELATED_RESULTS_ENABLED", "true")
    monkeypatch.setattr(
        verifier,
        "understand_question",
        lambda *args: UnderstoodQuestion(
            "lessons related to exercises",
            "mathematics",
            "EXERCISE",
            search_mode="browse",
            resource_format="lesson",
        ),
    )
    observed = []

    def retrieve_stub(_ws, question, *_args, **kwargs):
        observed.append((question, kwargs["content_type"], kwargs["subject"]))
        return [MATH_DOC]

    monkeypatch.setattr(verifier, "retrieve", retrieve_stub)
    ws = fake_workspace({"related": True, "reason": "Exercises are related resources"})
    response = verifier.verified_search(
        ws, "Can you show me lessons related to exercises?", "mathematics"
    )
    assert observed == [("exercise", "EXERCISE", "mathematics")]
    assert response["match_status"] == "related_results"
    assert response["related_results"][0]["content_type"] == "EXERCISE"


def test_related_search_broadens_candidate_generation_when_empty(monkeypatch):
    from att_agent.education import relevance_verifier as verifier

    monkeypatch.setenv("EDUCATION_QUERY_NORMALIZATION_ENABLED", "false")
    monkeypatch.setenv("EDUCATION_RELATED_RESULTS_ENABLED", "true")
    observed = []

    def retrieve_stub(_ws, question, *_args, **_kwargs):
        observed.append(question)
        return [MATH_DOC] if question == "exercise" else []

    monkeypatch.setattr(verifier, "retrieve", retrieve_stub)
    ws = fake_workspace({"related": True, "reason": "An exercise resource"})
    response = verifier.verified_search(
        ws, "Can you show me lessons related to exercises?", "mathematics"
    )
    assert observed == ["Can you show me lessons related to exercises?", "exercise"]
    assert response["related_results"][0]["search_id"] == "math-1"
