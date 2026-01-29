import uuid
from typing import Callable

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
    - Retention Score: How important is this information to remember?
    - Erasure Score: How safe is it to forget this information?
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

        prompt = f"""Analyze the following text and break it into meaningful memory chunks.
For each chunk, assign:
- retention_score (0.0-1.0): How important is this to remember? (1.0 = critical fact, 0.0 = trivial detail)
- erasure_score (0.0-1.0): How safe is it to forget? (1.0 = safe to forget, 0.0 = must never forget)

Guidelines:
- Key facts, names, dates, decisions → high retention, low erasure
- Background context, filler words → low retention, high erasure
- Assign unique IDs to each chunk

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

    def invoke(self, text: str) -> list[MemoryChunk]:
        """Process text and return committed memories."""
        initial_state: ERASEState = {
            'input_text': text,
            'chunks': [],
            'committed_memory': [],
            'iteration': 0
        }
        result = self._graph.invoke(initial_state)
        return result['committed_memory']

    def __call__(self, text: str) -> list[MemoryChunk]:
        return self.invoke(text)
