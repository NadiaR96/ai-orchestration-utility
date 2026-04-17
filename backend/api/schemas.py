from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class CompareRequest(BaseModel):
    input: str = Field(..., min_length=1)
    models: List[str] = Field(default_factory=lambda: ["small"])
    strategy: str = Field(default="balanced")


class RunRequest(BaseModel):
    input: str = Field(..., min_length=1)
    model: str = Field(default="small")
    strategy: str = Field(default="balanced")


class LeaderboardPromptRequest(BaseModel):
    input: str = Field(..., min_length=1)
    reference: Optional[str] = None
    models: List[str] = Field(default_factory=lambda: ["small", "default", "quality"])
    retrieval: str = Field(default="rag")
    sort_strategy: str = Field(default="balanced")
    aggregation: Literal["latest"] = "latest"
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)


class LeaderboardHistoryQuery(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)
    sort_strategy: str = Field(default="balanced")
    aggregation: Literal["latest"] = "latest"
    models: Optional[List[str]] = None