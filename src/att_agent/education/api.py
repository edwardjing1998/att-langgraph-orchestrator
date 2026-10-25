"""Education search API; legacy prototype routes remain unchanged."""

import asyncio
import logging
import os
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from att_agent.education.databricks_client import get_education_workspace_client
from att_agent.education.relevance_verifier import verified_search

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/education", tags=["education"])


class EducationSearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    subject: str = Field(default="", max_length=100)
    top_k: int = Field(default=10, ge=1, le=30)
    search_intent: Literal["auto", "answer_question", "find_resources"] = "auto"


@router.post("/search")
async def search(
    payload: EducationSearchRequest,
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    settings = request.app.state.settings
    if settings.app_api_key and x_api_key != settings.app_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    index = os.getenv("EDUCATION_SEARCH_INDEX", "education_rag.gold.approved_content_index")
    chat_endpoint = os.getenv("EDUCATION_CHAT_ENDPOINT", "databricks-gpt-oss-120b")
    try:
        ws = get_education_workspace_client()
        return await asyncio.to_thread(
            verified_search,
            ws,
            question=payload.question,
            subject=payload.subject,
            top_k=payload.top_k,
            index=index,
            chat_endpoint=chat_endpoint,
            search_intent=payload.search_intent,
        )
    except Exception as exc:
        LOGGER.exception("Education search failed")
        raise HTTPException(
            status_code=502,
            detail="Education retrieval unavailable",
        ) from exc
