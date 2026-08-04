from urllib.parse import quote

import httpx

from att_agent.models import Evidence


class SnowflakeCortexClient:
    def __init__(
        self,
        *,
        host: str,
        pat: str,
        database: str,
        schema: str,
        service: str,
        timeout_seconds: float,
    ) -> None:
        path = (
            "/api/v2/databases/"
            f"{quote(database, safe='')}/schemas/{quote(schema, safe='')}"
            f"/cortex-search-services/{quote(service, safe='')}:query"
        )
        self._path = path
        self._client = httpx.AsyncClient(
            base_url=host.rstrip("/"),
            timeout=timeout_seconds,
            headers={
                "Authorization": f"Bearer {pat}",
                "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def search(
        self,
        *,
        question: str,
        billing_account_token: str | None,
        limit: int,
    ) -> list[Evidence]:
        columns = [
            "DOCUMENT_ID",
            "BILL_ID",
            "BILLING_ACCOUNT_TOKEN",
            "CHARGE_TYPE_CODE",
            "CHARGE_STATUS",
            "SEARCHABLE_TEXT",
            "SOURCE_REFERENCE",
            "ACCESS_LEVEL",
        ]
        public_filter: dict[str, object] = {
            "@not": {"@eq": {"ACCESS_LEVEL": "RESTRICTED"}}
        }
        query_filter: dict[str, object] = public_filter
        if billing_account_token:
            query_filter = {
                "@and": [
                    {"@eq": {"BILLING_ACCOUNT_TOKEN": billing_account_token}},
                    public_filter,
                ]
            }

        response = await self._client.post(
            self._path,
            json={
                "query": question,
                "columns": columns,
                "filter": query_filter,
                "limit": limit,
            },
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("results", [])
        if not isinstance(rows, list):
            rows = payload.get("result", {}).get("results", [])

        evidence: list[Evidence] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized = {str(key).upper(): value for key, value in row.items()}
            evidence.append(
                Evidence(
                    source_platform="snowflake",
                    source_type="billing_transaction",
                    document_id=str(normalized.get("DOCUMENT_ID", "unknown")),
                    title=f"Billing record {normalized.get('BILL_ID', '')}".strip(),
                    content=str(normalized.get("SEARCHABLE_TEXT", "")),
                    source_reference=_optional_text(normalized.get("SOURCE_REFERENCE")),
                    access_level=_optional_text(normalized.get("ACCESS_LEVEL")),
                    score=_optional_float(normalized.get("_SCORE", normalized.get("SCORE"))),
                    metadata={
                        "bill_id": normalized.get("BILL_ID"),
                        "billing_account_token": normalized.get("BILLING_ACCOUNT_TOKEN"),
                        "charge_type_code": normalized.get("CHARGE_TYPE_CODE"),
                        "charge_status": normalized.get("CHARGE_STATUS"),
                    },
                )
            )
        return evidence


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None

