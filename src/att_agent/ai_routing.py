import json
import logging

import httpx
from pydantic import BaseModel, Field, ValidationError

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
    ) -> None:
        self._endpoint = endpoint

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
        try:
            payload = await self._call_model(question)
            decision = _parse_decision(payload)

            if not decision.intents:
                raise ValueError(
                    "The AI router returned no intents"
                )

            return decision

        except (
            httpx.HTTPError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
            ValidationError,
        ) as exception:
            LOGGER.warning(
                "AI routing failed; using deterministic fallback: %s",
                type(exception).__name__,
            )

            fallback_intents = classify_intents(question)

            return RoutingDecision(
                intents=fallback_intents,
                needs_billing_account_token=(
                    "billing_transaction"
                    in fallback_intents
                ),
                reason="Deterministic routing fallback",
            )

    async def _call_model(
        self,
        question: str,
    ) -> dict[str, object]:
        system_prompt = """
You route telecom questions to approved retrieval tools.

Available intents:

1. policy
Use for:
- policy rules
- refund or reversal eligibility
- disputes
- what is allowed
- product or service policies

2. billing_knowledge
Use for:
- general billing explanations
- fees
- taxes
- proration
- roaming-charge explanations
- meanings of charge types

3. billing_transaction
Use for:
- the user's bill
- customer-specific charge amounts
- transactions
- credits
- adjustments
- billing-cycle records
- posted charge status

A question may require multiple intents.

Return only valid JSON in this exact form:

{
  "intents": [
    "policy",
    "billing_knowledge",
    "billing_transaction"
  ],
  "needs_billing_account_token": true,
  "reason": "Short non-sensitive explanation"
}

Rules:
- Select only the listed intents.
- Do not generate URLs.
- Do not request credentials.
- Do not include hidden reasoning.
- Never infer customer identity.
- billing_transaction requires a billing-account token.
- General policy and knowledge questions do not require customer data.
""".strip()

        response = await self._client.post(
            (
                f"/serving-endpoints/"
                f"{self._endpoint}/invocations"
            ),
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

    # Remove duplicates while preserving order.
    decoded["intents"] = list(
        dict.fromkeys(decoded.get("intents", []))
    )

    return RoutingDecision.model_validate(decoded)


def _extract_generated_text(
    payload: dict[str, object],
) -> str:
    choices = payload.get("choices")

    if isinstance(choices, list) and choices:
        first = choices[0]

        if isinstance(first, dict):
            message = first.get("message")

            if isinstance(message, dict):
                content = message.get("content")

                if content:
                    return str(content)

            text = first.get("text")

            if text:
                return str(text)

    predictions = payload.get("predictions")

    if isinstance(predictions, list) and predictions:
        first = predictions[0]

        if isinstance(first, dict):
            for key in (
                "content",
                "text",
                "generated_text",
            ):
                value = first.get(key)

                if value:
                    return str(value)

        return str(first)

    raise ValueError(
        "The routing model returned no generated text"
    )


def _remove_code_fence(value: str) -> str:
    stripped = value.strip()

    if stripped.startswith("```"):
        lines = stripped.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        stripped = "\n".join(lines).strip()

    return stripped