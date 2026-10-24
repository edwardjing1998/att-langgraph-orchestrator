import json
import logging

import httpx
from pydantic import BaseModel, Field

from att_agent.models import Intent
from att_agent.routing import classify_intents

LOGGER = logging.getLogger(__name__)


class RoutingDecision(BaseModel):
    intents: list[Intent] = Field(min_length=1)
    needs_billing_account_token: bool = False
    reason: str = Field(max_length=500)


class DatabricksAIRouter:
    def __init__(
        self,
        *,
        host: str,
        token: str,
        endpoint: str,
        timeout_seconds: float,
        enabled: bool,
    ) -> None:
        self._endpoint = endpoint
        self._enabled = enabled

        self._client = httpx.AsyncClient(
            base_url=host.rstrip("/"),
            timeout=timeout_seconds,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def classify(
        self,
        question: str,
    ) -> RoutingDecision:
        if not self._enabled:
            return _fallback_decision(question)

        try:
            payload = await self._call_model(question)
            return _parse_decision(payload)

        except (
            httpx.HTTPError,
            ValueError,
            KeyError,
            TypeError,
        ) as exception:
            LOGGER.warning(
                "AI routing failed; using deterministic fallback: %s",
                type(exception).__name__,
            )

            return _fallback_decision(question)

    async def _call_model(
        self,
        question: str,
    ) -> dict[str, object]:
        system_prompt = """
You route telecom questions to approved retrieval tools.

Allowed intents:

- policy:
  Policy rules, eligibility, refunds, reversals, and disputes.

- billing_knowledge:
  General explanations of fees, taxes, proration, roaming,
  and charge types.

- billing_transaction:
  Customer-specific bills, amounts, transactions, credits,
  adjustments, billing cycles, and posting status.

A question can require multiple intents.

Return only JSON with this shape:

{
  "intents": [
    "policy",
    "billing_knowledge",
    "billing_transaction"
  ],
  "needs_billing_account_token": true,
  "reason": "Short, non-sensitive explanation"
}

Use only the allowed intent names.

Do not generate URLs, credentials, SQL, or hidden reasoning.
Never infer customer identity.
Customer-specific transaction retrieval requires a billing-account token.
""".strip()

        response = await self._client.post(
            (f"/serving-endpoints/{self._endpoint}/invocations"),
            json={
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "temperature": 0,
                "max_tokens": 250,
            },
        )

        response.raise_for_status()

        return response.json()


def _parse_decision(
    payload: dict[str, object],
) -> RoutingDecision:
    generated_text = _extract_generated_text(payload)
    cleaned_text = _remove_code_fence(generated_text)
    decoded = json.loads(cleaned_text)

    if not isinstance(decoded, dict):
        raise ValueError("The routing model response must be a JSON object")

    intents = decoded.get("intents", [])

    if isinstance(intents, list):
        # Remove duplicate intents while preserving order.
        decoded["intents"] = list(dict.fromkeys(intents))

    return RoutingDecision.model_validate(decoded)


def _extract_generated_text(
    payload: dict[str, object],
) -> str:
    choices = payload.get("choices")

    if isinstance(choices, list) and choices:
        first = choices[0]

        if isinstance(first, dict):
            message = first.get("message")

            if isinstance(message, dict) and message.get("content"):
                return str(message["content"])

            if first.get("text"):
                return str(first["text"])

    predictions = payload.get("predictions")

    if isinstance(predictions, list) and predictions:
        first = predictions[0]

        if isinstance(first, dict):
            for key in (
                "content",
                "text",
                "generated_text",
            ):
                if first.get(key):
                    return str(first[key])

        return str(first)

    raise ValueError("The routing model returned no generated text")


def _remove_code_fence(
    value: str,
) -> str:
    stripped = value.strip()

    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()[1:]

    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines).strip()


def _fallback_decision(
    question: str,
) -> RoutingDecision:
    intents = classify_intents(question)

    return RoutingDecision(
        intents=intents,
        needs_billing_account_token=("billing_transaction" in intents),
        reason="Deterministic routing fallback",
    )
