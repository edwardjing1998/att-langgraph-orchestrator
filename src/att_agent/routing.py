import re

from att_agent.models import Intent

POLICY_TERMS = {
    "policy",
    "eligible",
    "eligibility",
    "allowed",
    "reversed",
    "reverse",
    "refund",
    "dispute",
    "rule",
}

BILLING_KNOWLEDGE_TERMS = {
    "explain",
    "meaning",
    "mean",
    "prorated",
    "proration",
    "fee",
    "tax",
    "day-pass",
    "day pass",
    "charge type",
}

TRANSACTION_TERMS = {
    "my bill",
    "my charge",
    "charged",
    "transaction",
    "record",
    "amount",
    "credit",
    "adjustment",
    "posted",
    "billing cycle",
    "$",
}


def classify_intents(question: str) -> list[Intent]:
    normalized = re.sub(r"\s+", " ", question.lower()).strip()
    intents: list[Intent] = []

    if any(term in normalized for term in POLICY_TERMS):
        intents.append("policy")
    if any(term in normalized for term in BILLING_KNOWLEDGE_TERMS):
        intents.append("billing_knowledge")
    if any(term in normalized for term in TRANSACTION_TERMS):
        intents.append("billing_transaction")

    # Roaming questions often need a billing explanation even when the user does
    # not explicitly say "explain".
    if "roaming" in normalized and "billing_knowledge" not in intents:
        intents.append("billing_knowledge")

    # A safe default for vague questions is knowledge retrieval, not customer
    # transaction retrieval.
    if not intents:
        intents.append("billing_knowledge")

    return intents

