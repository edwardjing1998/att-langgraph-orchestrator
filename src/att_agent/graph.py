import asyncio
from collections.abc import Awaitable

import httpx
from langgraph.graph import END, START, StateGraph

from att_agent.ai_routing import DatabricksAIRouter
from att_agent.clients.databricks import DatabricksClient
from att_agent.clients.snowflake import SnowflakeCortexClient
from att_agent.config import Settings
from att_agent.models import AgentState, Evidence, Intent


class AgentGraph:
    def __init__(
        self,
        settings: Settings,
        databricks: DatabricksClient,
        snowflake: SnowflakeCortexClient,
        ai_router: DatabricksAIRouter,
    ) -> None:
        self._settings = settings
        self._databricks = databricks
        self._snowflake = snowflake
        self._ai_router = ai_router

        builder = StateGraph(AgentState)

        builder.add_node(
            "classify",
            self._classify,
        )

        builder.add_node(
            "retrieve",
            self._retrieve,
        )

        builder.add_node(
            "synthesize",
            self._synthesize,
        )

        builder.add_edge(
            START,
            "classify",
        )

        builder.add_edge(
            "classify",
            "retrieve",
        )

        builder.add_edge(
            "retrieve",
            "synthesize",
        )

        builder.add_edge(
            "synthesize",
            END,
        )

        self.compiled = builder.compile()

    async def _classify(
        self,
        state: AgentState,
    ) -> dict[str, object]:
        decision = await self._ai_router.classify(state["question"])

        return {
            "intents": decision.intents,
        }

    async def _retrieve(
        self,
        state: AgentState,
    ) -> dict[str, object]:
        intents = state["intents"]
        limit = state["max_results_per_source"]

        tasks: list[
            tuple[
                Intent,
                Awaitable[list[Evidence]],
            ]
        ] = []

        errors: list[str] = []

        if "policy" in intents:
            tasks.append(
                (
                    "policy",
                    self._databricks.search(
                        index_name=(self._settings.databricks_policy_index),
                        question=state["question"],
                        source_type="policy",
                        limit=limit,
                    ),
                )
            )

        if "billing_knowledge" in intents:
            tasks.append(
                (
                    "billing_knowledge",
                    self._databricks.search(
                        index_name=(self._settings.databricks_billing_index),
                        question=state["question"],
                        source_type="billing_knowledge",
                        limit=limit,
                    ),
                )
            )

        if "billing_transaction" in intents:
            billing_account_token = state.get("billing_account_token")

            if billing_account_token:
                tasks.append(
                    (
                        "billing_transaction",
                        self._snowflake.search(
                            question=state["question"],
                            billing_account_token=(billing_account_token),
                            limit=limit,
                        ),
                    )
                )
            else:
                errors.append("billing_transaction requires an authorized billing account token")

        results = await asyncio.gather(
            *(task for _, task in tasks),
            return_exceptions=True,
        )

        evidence: list[Evidence] = []

        for (
            intent,
            _,
        ), result in zip(
            tasks,
            results,
            strict=True,
        ):
            if isinstance(result, Exception):
                errors.append(
                    _safe_error(
                        intent,
                        result,
                    )
                )
            else:
                evidence.extend(result)

        return {
            "evidence": evidence,
            "errors": errors,
        }

    async def _synthesize(
        self,
        state: AgentState,
    ) -> dict[str, object]:
        evidence = state.get(
            "evidence",
            [],
        )

        if not evidence:
            return {
                "answer": (
                    "I could not retrieve approved evidence "
                    "for this question. Please try a more "
                    "specific question or contact support."
                )
            }

        if self._settings.llm_enabled:
            try:
                answer = await self._databricks.generate_answer(
                    state["question"],
                    evidence,
                )

                return {
                    "answer": answer,
                }

            except Exception as exception:
                errors = [
                    *state.get(
                        "errors",
                        [],
                    ),
                    _safe_error(
                        "answer_generation",
                        exception,
                    ),
                ]

                return {
                    "answer": _extractive_fallback(evidence),
                    "errors": errors,
                }

        return {"answer": _extractive_fallback(evidence)}

    async def close(self) -> None:
        await asyncio.gather(
            self._databricks.close(),
            self._snowflake.close(),
            self._ai_router.close(),
        )


def build_graph(
    settings: Settings,
) -> AgentGraph:
    databricks = DatabricksClient(
        host=settings.databricks_host,
        token=settings.databricks_token,
        chat_endpoint=(settings.databricks_chat_endpoint),
        timeout_seconds=(settings.request_timeout_seconds),
    )

    snowflake = SnowflakeCortexClient(
        host=settings.snowflake_host,
        pat=settings.snowflake_pat,
        database=(settings.snowflake_cortex_database),
        schema=(settings.snowflake_cortex_schema),
        service=(settings.snowflake_cortex_service),
        timeout_seconds=(settings.request_timeout_seconds),
    )

    ai_router = DatabricksAIRouter(
        host=settings.databricks_host,
        token=settings.databricks_token,
        endpoint=(settings.databricks_chat_endpoint),
        timeout_seconds=(settings.request_timeout_seconds),
        enabled=settings.llm_enabled,
    )

    return AgentGraph(
        settings=settings,
        databricks=databricks,
        snowflake=snowflake,
        ai_router=ai_router,
    )


def _extractive_fallback(
    evidence: list[Evidence],
) -> str:
    lines = ["Retrieved approved evidence:"]

    for item in evidence[:6]:
        compact = " ".join(item.content.split())

        if len(compact) > 280:
            compact = f"{compact[:277]}..."

        lines.append(f"- [{item.document_id}] {compact}")

    return "\n".join(lines)


def _safe_error(
    source: str,
    exception: BaseException,
) -> str:
    if isinstance(
        exception,
        httpx.HTTPStatusError,
    ):
        return f"{source} returned HTTP {exception.response.status_code}"

    if isinstance(
        exception,
        httpx.TimeoutException,
    ):
        return f"{source} timed out"

    return f"{source} failed: {type(exception).__name__}"
