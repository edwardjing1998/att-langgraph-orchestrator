"""Image metadata should survive retrieval only for verified search results."""

import json
from types import SimpleNamespace

from att_agent.education.image_resolver import resolve_document_images
from att_agent.education.relevance_verifier import retrieve, verify_one


def test_absolute_urls_from_gold_metadata():
    doc = {"image_links_json": json.dumps(["https://images.example.org/a.png"])}
    assert resolve_document_images(doc)[0]["url"] == "https://images.example.org/a.png"


def test_relative_paths_require_explicit_base():
    doc = {"image_links_json": '["book/chapter/figure-1.png"]'}
    assert resolve_document_images(doc, base_url="")[0]["url"] is None
    assert (
        resolve_document_images(doc, base_url="https://storage.example.org/education")[0]["url"]
        == "https://storage.example.org/education/book/chapter/figure-1.png"
    )


def test_duplicate_images_and_malformed_metadata():
    assert resolve_document_images({"image_links_json": "oops"}) == []
    duplicate_images = {
        "image_links_json": ('["https://a.example/x.png", "https://a.example/x.png"]')
    }
    assert len(resolve_document_images(duplicate_images)) == 1


def test_no_path_traversal():
    doc = {"image_links_json": '["../private.png"]'}
    assert resolve_document_images(doc, base_url="https://host.example/images")[0]["url"] is None


def test_vector_search_requests_image_metadata():
    class FakeClient:
        def do(self, **kwargs):
            assert "image_links_json" in kwargs["body"]["columns"]
            return {
                "manifest": {
                    "columns": [
                        {"name": "search_id"},
                        {"name": "subject"},
                        {"name": "image_links_json"},
                    ]
                },
                "result": {
                    "data_array": [["one", "bible", '["https://images.example.org/a.png"]']]
                },
            }

    ws = SimpleNamespace(api_client=FakeClient())
    docs = retrieve(ws, "Abraham", 10, subject="bible")
    assert len(docs) == 1
    assert docs[0]["image_links_json"]


def test_verified_result_keeps_image_url():
    choice = SimpleNamespace(
        message=SimpleNamespace(
            content=json.dumps(
                {
                    "relevant": True,
                    "evidence": "A right triangle has legs 6 cm and 8 cm.",
                    "reason": "Matching exercise",
                    "match_type": "EXERCISE_MATCH",
                    "answer_supported": False,
                }
            )
        )
    )
    ws = SimpleNamespace(
        serving_endpoints=SimpleNamespace(query=lambda **kwargs: SimpleNamespace(choices=[choice]))
    )
    doc = {
        "search_id": "one",
        "subject": "mathematics",
        "content_type": "EXERCISE",
        "searchable_text": "A right triangle has legs 6 cm and 8 cm. Find x.",
        "image_links_json": '["https://images.example.org/triangle.png"]',
    }
    result, reason = verify_one(ws, "Find a triangle exercise", doc, "model")
    assert reason == "accepted"
    assert result["images"][0]["url"] == "https://images.example.org/triangle.png"


def test_accepted_image_metadata_absent_is_empty():
    assert resolve_document_images({}) == []
