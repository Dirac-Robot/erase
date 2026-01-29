from typing import Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class MemoryChunk(BaseModel):
    """A scored memory unit with retention and erasure scores."""
    id: str = Field(description='Unique identifier for this memory chunk')
    content: str = Field(description='The actual content of this memory')
    retention_score: float = Field(
        ge=0.0, le=1.0,
        description='How important this information is to remember (0=unimportant, 1=critical)'
    )
    erasure_score: float = Field(
        ge=0.0, le=1.0,
        description='How safe it is to forget this information (0=must keep, 1=safe to forget)'
    )
    metadata: dict = Field(default_factory=dict)


class ScoredChunks(BaseModel):
    """LLM output schema for scoring text chunks."""
    chunks: list[MemoryChunk] = Field(description='List of scored memory chunks')
    reasoning: Optional[str] = Field(
        default=None,
        description='Brief explanation of scoring rationale'
    )


class ERASEState(TypedDict):
    """Main state for ERASE workflow."""
    input_text: str
    chunks: list[MemoryChunk]
    committed_memory: list[MemoryChunk]
    iteration: int
