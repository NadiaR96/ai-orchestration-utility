from pydantic import BaseModel, Field
from typing import List, Optional


class CompareRequest(BaseModel):
    input: str = Field(..., min_length=1)
    models: List[str] = Field(default_factory=lambda: ["small"])
    strategy: str = Field(default="balanced")


class RunRequest(BaseModel):
    input: str = Field(..., min_length=1)
    model: str = Field(default="small")
    strategy: str = Field(default="balanced")