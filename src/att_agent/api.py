import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from att_agent.education.api import router as education_router
from att_agent.config import Settings, get_settings
from att_agent.graph import AgentGraph, build_graph
from att_agent.models import AgentState, AskRequest, AskResponse

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    app.state.settings = settings
    app.state.graph = build_graph(settings)
    yield
    await app.state.graph.close()


app = FastAPI(
    title="AT&T-style LangGraph AI Search Orchestrator",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the React application to call this API from a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://admin-panel-edward-jing-dev.apps.rm1.0a51.p1.openshiftapps.com",
        "https://admin-panel-go4share-me-dev.apps.rm2.thpm.p1.openshiftapps.com",
        "http://localhost:3000",
    ],
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    settings: Settings = request.app.state.settings

    if settings.app_api_key and x_api_key != settings.app_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "UP"}


@app.get("/health/ready")
async def ready(request: Request) -> dict[str, str]:
    graph = getattr(request.app.state, "graph", None)

    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph is not initialized",
        )

    return {"status": "UP"}


@app.post(
    "/api/v1/ask",
    response_model=AskResponse,
    dependencies=[Depends(require_api_key)],
)
async def ask(
    request_body: AskRequest,
    request: Request,
) -> AskResponse:
    settings: Settings = request.app.state.settings
    graph: AgentGraph = request.app.state.graph
    trace_id = str(uuid.uuid4())

    initial_state: AgentState = {
        "trace_id": trace_id,
        "question": request_body.question,
        "billing_account_token": request_body.billing_account_token,
        "max_results_per_source": (
            request_body.max_results_per_source
            or settings.max_results_per_source
        ),
        "errors": [],
        "evidence": [],
    }

    LOGGER.info("agent request trace_id=%s", trace_id)

    result = await graph.compiled.ainvoke(initial_state)

    return AskResponse(
        traceId=trace_id,
        question=request_body.question,
        intents=result.get("intents", []),
        answer=result.get("answer", "No answer was generated."),
        evidence=result.get("evidence", []),
        errors=result.get("errors", []),
    )

# Isolated education retrieval route; existing prototype endpoints unchanged.
app.include_router(education_router)
