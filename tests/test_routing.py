import pytest

from att_agent.routing import classify_intents


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Can an incorrect roaming charge be reversed?", {"policy", "billing_knowledge"}),
        ("What does a prorated service charge mean?", {"billing_knowledge"}),
        ("Show the amount charged on my bill", {"billing_transaction"}),
        (
            "I was charged $24 for roaming. Explain it and tell me whether it can be reversed.",
            {"policy", "billing_knowledge", "billing_transaction"},
        ),
    ],
)
def test_classify_intents(question: str, expected: set[str]) -> None:
    assert set(classify_intents(question)) == expected


def test_unknown_question_defaults_to_billing_knowledge() -> None:
    assert classify_intents("Please help me understand this") == ["billing_knowledge"]

