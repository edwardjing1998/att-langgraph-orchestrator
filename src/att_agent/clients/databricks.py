import json

import httpx

from att_agent.models import Evidence, Intent


class DatabricksClient:
    def __init__(
        self,
        host: str,
        token: str,
        chat_endpoint: str,
        timeout_seconds: float,
    ) -> None:
        self._host = host.rstrip("/")
        self._chat_endpoint = chat_endpoint

        self._client = httpx.AsyncClient(
            base_url=self._host,
            timeout=timeout_seconds,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def search(
        self,
        *,
        index_name: str,
        question: str,
        source_type: Intent,
        limit: int,
    ) -> list[Evidence]:
        columns = [
            "document_id",
            "title",
            "domain",
            "category",
            "searchable_text",
            "source_reference",
            "access_level",
        ]

        response = await self._client.request(
            method="GET",
            url=(f"/api/2.0/vector-search/indexes/{index_name}/query"),
            json={
                "query_text": question,
                "columns": columns,
                "filters": {"access_level NOT": "RESTRICTED"},
                "num_results": limit,
                "query_type": "HYBRID",
            },
        )

        response.raise_for_status()

        payload = response.json()
        rows = payload.get("result", {}).get("data_array", [])

        evidence: list[Evidence] = []

        for row in rows:
            if len(row) < 7:
                continue

            evidence.append(
                Evidence(
                    source_platform="databricks",
                    source_type=source_type,
                    document_id=str(row[0]),
                    title=_optional_text(row[1]),
                    content=str(row[4] or ""),
                    source_reference=_optional_text(row[5]),
                    access_level=_optional_text(row[6]),
                    score=(float(row[7]) if len(row) > 7 and row[7] is not None else None),
                    metadata={
                        "domain": row[2],
                        "category": row[3],
                    },
                )
            )

        return evidence

    async def generate_answer(
        self,
        question: str,
        evidence: list[Evidence],
    ) -> str:
        context = [item.model_dump(mode="json") for item in evidence]

        prompt = (
            "Answer the user using only the supplied evidence. "
            "Clearly distinguish confirmed billing facts from policy guidance. "
            "If evidence is insufficient, say so. "
            "Cite sources inline using [document_id] markers. "
            "Never expose credentials or infer customer PII.\n\n"
            f"Question:\n{question}\n\n"
            f"Evidence:\n{json.dumps(context, indent=2)}"
        )

        response = await self._client.post(
            (f"/serving-endpoints/{self._chat_endpoint}/invocations"),
            json={
                "messages": [
                    {
                        "role": "system",
                        "content": ("You are a governed telecom billing assistant."),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "max_tokens": 700,
                "temperature": 0.1,
            },
        )

        response.raise_for_status()

        return _extract_generated_text(response.json())


def _optional_text(value: object) -> str | None:
    if value is None:
        return None

    return str(value)


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

    raise ValueError("Databricks chat endpoint returned no generated answer")
