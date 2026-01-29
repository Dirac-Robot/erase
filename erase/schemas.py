from datetime import datetime
from typing import Optional, Literal
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
        description='How much this should be excluded (0=keep, 1=exclude). High score means either trivial OR must-be-excluded from current context.'
    )
    category: Optional[str] = Field(default=None, description='Optional category label')


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
    query: Optional[str]
    chunks: list[MemoryChunk]
    committed_memory: list[MemoryChunk]
    iteration: int


class Message(BaseModel):
    """A conversation message with optional ERASE scores."""
    role: Literal['user', 'assistant', 'system'] = Field(description='Message role')
    content: str = Field(description='Message content')
    timestamp: datetime = Field(default_factory=datetime.now)
    retention_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    erasure_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    def to_str(self) -> str:
        return f"[{self.role}] {self.content}"
