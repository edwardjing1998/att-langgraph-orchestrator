from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

Intent = Literal["policy", "billing_knowledge", "billing_transaction"]


class AskRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(min_length=3, max_length=2000)
    billing_account_token: str | None = Field(
        default=None,
        alias="billingAccountToken",
        max_length=200,
    )
    max_results_per_source: int | None = Field(
        default=None,
        alias="maxResultsPerSource",
        ge=1,
        le=10,
    )


class Evidence(BaseModel):
    source_platform: Literal["databricks", "snowflake"]
    source_type: Intent
    document_id: str
    title: str | None = None
    content: str
    source_reference: str | None = None
    access_level: str | None = None
    score: float | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class AskResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    trace_id: str = Field(alias="traceId")
    question: str
    intents: list[Intent]
    answer: str
    evidence: list[Evidence]
    errors: list[str]


class AgentState(TypedDict, total=False):
    trace_id: str
    question: str
    billing_account_token: str | None
    max_results_per_source: int
    intents: list[Intent]
    evidence: list[Evidence]
    errors: list[str]
    answer: str
