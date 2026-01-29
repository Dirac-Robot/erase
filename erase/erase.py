import uuid
from typing import Optional

from ato.adict import ADict
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from erase.schemas import ERASEState, MemoryChunk, ScoredChunks


class ERASE:
    """
    Explicit Retention And Selective Erasure

    A dual-scored memory system that explicitly models:
    - Retention Score: How important is this information?
    - Erasure Score: Should this be excluded? (trivial OR must-exclude)
    
    Both scores are independent (don't sum to 1).
    High erasure overrides high retention.
    
    Key insight: Scores are QUERY-CONDITIONAL.
    The same chunk may have different erasure scores for different queries.
    """

    def __init__(self, config: ADict):
        self._config = config
        self._llm: BaseChatModel = None
        self._graph = None
        self._build()

    def _build(self):
        self._llm = ChatOpenAI(model=self._config.model)

        graph = StateGraph(ERASEState)
        graph.add_node('chunk_and_score', self.chunk_and_score)
        graph.add_node('apply_threshold', self.apply_threshold)

        graph.add_edge(START, 'chunk_and_score')
        graph.add_edge('chunk_and_score', 'apply_threshold')
        graph.add_edge('apply_threshold', END)

        self._graph = graph.compile()

    @property
    def config(self) -> ADict:
        return self._config

    @property
    def graph(self):
        return self._graph

    def chunk_and_score(self, state: ERASEState) -> dict:
        """Chunk input text and score each chunk for retention and erasure."""
        structured_llm = self._llm.with_structured_output(ScoredChunks)

        query = state.get('query')
        
        if query:
            prompt = f"""You are a memory scoring system. Analyze the text and score each chunk.

## SCORING RULES

### retention_score (0.0-1.0): General Importance
How important is this information IN ISOLATION, regardless of any query?
- 1.0: Critical facts (names, numbers, dates, decisions, key data)
- 0.7: Important but not critical
- 0.4: Moderately useful
- 0.0: Trivial (greetings, filler, small talk)

### erasure_score (0.0-1.0): Query Relevance (CRITICAL!)
Should this be EXCLUDED when answering the specific query below?

**CURRENT QUERY: "{query}"**

Think: "Does this chunk DIRECTLY help answer the query?"
- 0.0: Directly relevant to the query (KEEP)
- 0.3: Somewhat related (probably keep)
- 0.7: Different topic but important (EXCLUDE despite importance)
- 1.0: Completely off-topic or noise (EXCLUDE)

## KEY INSIGHT
A chunk can have HIGH retention (important info) AND HIGH erasure (off-topic for THIS query).
Example: "Project A costs $1M" has high retention, but if query asks about Project B, erasure should be HIGH.

## EXAMPLES
Query: "What is Project B's budget?"
- "Project B budget is $500K" → R=1.0, E=0.0 (directly answers)
- "Project A budget is $1M" → R=0.9, E=0.9 (important but wrong project!)
- "Let's grab coffee" → R=0.1, E=1.0 (trivial and off-topic)

## TEXT TO ANALYZE:
{state['input_text']}

Break into meaningful chunks and score each. Use unique IDs like chunk1, chunk2, etc."""
        else:
            prompt = f"""Analyze the following text and break it into meaningful memory chunks.
For each chunk, assign:

- retention_score (0.0-1.0): How important is this information?
  - 1.0 = critical fact (names, dates, decisions, key numbers)
  - 0.0 = trivial detail

- erasure_score (0.0-1.0): Should this be excluded?
  - Use 0.0 for most chunks (no specific query to filter by)
  - Use higher scores only for sensitive or clearly temporary info

Assign unique IDs to each chunk.

Text to analyze:
{state['input_text']}"""

        result: ScoredChunks = structured_llm.invoke(prompt)

        for chunk in result.chunks:
            if not chunk.id:
                chunk.id = str(uuid.uuid4())[:8]

        return {'chunks': result.chunks}

    def apply_threshold(self, state: ERASEState) -> dict:
        """Filter chunks based on retention and erasure thresholds."""
        retention_threshold = self._config.threshold.retention
        erasure_threshold = self._config.threshold.erasure

        committed = []
        for chunk in state['chunks']:
            keep = (
                chunk.retention_score >= retention_threshold and
                chunk.erasure_score < erasure_threshold
            )
            if keep:
                committed.append(chunk)

        return {
            'committed_memory': committed,
            'iteration': state.get('iteration', 0)+1
        }

    def invoke(self, text: str, query: Optional[str] = None) -> list[MemoryChunk]:
        """Process text and return committed memories (filtered by threshold).
        
        Args:
            text: The input text to analyze and chunk
            query: Optional query/context to condition the erasure scoring
        """
        initial_state: ERASEState = {
            'input_text': text,
            'query': query,
            'chunks': [],
            'committed_memory': [],
            'iteration': 0
        }
        result = self._graph.invoke(initial_state)
        return result['committed_memory']

    def score_all(self, text: str, query: Optional[str] = None) -> list[MemoryChunk]:
        """Score all chunks without threshold filtering.
        
        Use this to see scores for ALL chunks, including excluded ones.
        Useful for debugging and analysis.
        
        Args:
            text: The input text to analyze and chunk
            query: Optional query/context to condition the erasure scoring
        """
        initial_state: ERASEState = {
            'input_text': text,
            'query': query,
            'chunks': [],
            'committed_memory': [],
            'iteration': 0
        }
        result = self._graph.invoke(initial_state)
        return result['chunks']  # Return ALL chunks, not just committed

    def __call__(self, text: str, query: Optional[str] = None) -> list[MemoryChunk]:
        return self.invoke(text, query)

