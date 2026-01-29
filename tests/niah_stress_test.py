"""
NIAH Batch Processing - Binary ERASE Logic

Uses binary is_relevant/should_exclude instead of float scores.
"""
import asyncio
import re
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from ato.adict import ADict

from erase import scope


BATCH_SIZE = 15


class SentenceScore(BaseModel):
    """Binary score for a single sentence."""
    id: str = Field(description="Unique ID (s1, s2, etc)")
    content: str = Field(description="The sentence content")
    is_relevant: bool = Field(description="Is this related to the query?")
    should_exclude: bool = Field(description="Should this be EXCLUDED? True=related but doesn't answer")


class BatchScoreResult(BaseModel):
    """Batch scoring result."""
    chunks: list[SentenceScore]


def load_document() -> str:
    doc_path = Path(__file__).parent / "yuzi.txt"
    return doc_path.read_text(encoding='utf-8')


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?。다요죠])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def create_erase_prompt(sentences_text: str, query: str) -> str:
    """Simplified binary ERASE prompt with clear examples."""
    return f"""You are a document filter. For each sentence, decide:

### is_relevant
- true: Related to the query topic
- false: Completely unrelated

### should_exclude
- false (KEEP): Sentence doesn't have to be excluded.
- true (EXCLUDE): Sentence has to be excluded.

**QUERY: "{query}"**

## EXAMPLES
Query: "Who developed Linux?"
- "Linux was developed by Linus Torvalds in 1991." → is_relevant=true, should_exclude=false (KEEP: has the answer "Linus Torvalds")
- "Linux is open-source and free to use." → is_relevant=true, should_exclude=true (EXCLUDE: about Linux, but no answer to "Who")
- "Python is a programming language." → is_relevant=false, should_exclude=true (EXCLUDE: unrelated)

## SENTENCES:
{sentences_text}"""


def create_single_prompt(sentences_text: str, query: str) -> str:
    """Aligned Single prompt for fair comparison."""
    return f"""You are a high-precision document filter. Your goal is to identify relevant sentences.

## CLASSIFICATION RULES

### 1. is_relevant (boolean)
- **true**: The sentence mentions the same entities or topics as the query.
- **false**: Completely off-topic.

### 2. should_exclude (boolean)
- Always set to **false** (KEEP) for all sentences in this mode.

**QUERY: "{query}"**

## SENTENCES:
{sentences_text}"""


class BatchScorer:
    """Score batches using binary ERASE logic."""
    
    def __init__(self, config: ADict, use_erasure: bool = True):
        self._llm = ChatOpenAI(model=config.model)
        self._use_erasure = use_erasure
    
    def score_batch(self, sentences: list[str], query: str) -> list[SentenceScore]:
        """Score a batch of sentences."""
        structured_llm = self._llm.with_structured_output(BatchScoreResult)
        sentences_text = '\n'.join(f"[{i+1}] {s}" for i, s in enumerate(sentences))
        
        if self._use_erasure:
            prompt = create_erase_prompt(sentences_text, query)
        else:
            prompt = create_single_prompt(sentences_text, query)
        
        result: BatchScoreResult = structured_llm.invoke(prompt)
        return result.chunks
    
    def filter_batch(self, sentences: list[str], query: str) -> list[str]:
        """Score and filter a batch."""
        scores = self.score_batch(sentences, query)
        
        # Create a map for original content
        id_to_original = {f"s{i+1}": s for i, s in enumerate(sentences)}
        
        if self._use_erasure:
            # ERASE: keep if relevant AND not excluded
            kept_ids = [s.id for s in scores if s.is_relevant and not s.should_exclude]
        else:
            # Single: keep if relevant only
            kept_ids = [s.id for s in scores if s.is_relevant]
        
        return [id_to_original[cid] for cid in kept_ids if cid in id_to_original]


async def score_all_batches(scorer: BatchScorer, all_sentences: list[str], query: str) -> list[str]:
    """Score all batches in parallel."""
    batches = [all_sentences[i:i+BATCH_SIZE] for i in range(0, len(all_sentences), BATCH_SIZE)]
    
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, scorer.filter_batch, batch, query) for batch in batches]
    results = await asyncio.gather(*tasks)
    
    all_kept = []
    for batch_kept in results:
        all_kept.extend(batch_kept)
    return all_kept


def answer_question(llm, context: list[str], query: str) -> str:
    """Final LLM call to answer."""
    context_text = '\n'.join(context[:30])
    prompt = f"""Context:
{context_text}

Question: {query}
Answer (한국어로, 간결하게):"""
    return llm.invoke(prompt).content


@scope
def main(config):
    erase_scorer = BatchScorer(config, use_erasure=True)
    single_scorer = BatchScorer(config, use_erasure=False)
    llm = ChatOpenAI(model=config.model)
    
    print("=" * 70)
    print("NIAH Batch Processing - Binary ERASE")
    print("=" * 70)
    
    raw_doc = load_document()
    sentences = split_sentences(raw_doc)
    num_batches = (len(sentences) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\n{len(sentences)} sentences → {num_batches} batches × {BATCH_SIZE}")
    print()
    
    tests = [
        {"query": "고죠 사토루를 봉인한 사람은?", "answer": "켄자쿠"},
    ]
    
    async def run():
        for test in tests:
            query, expected = test['query'], test['answer']
            print(f"Q: {query}")
            print("-" * 50)
            
            single_kept = await score_all_batches(single_scorer, sentences, query)
            erase_kept = await score_all_batches(erase_scorer, sentences, query)
            
            print(f"  Single: {len(single_kept)}/{len(sentences)} kept")
            print(f"  ERASE:  {len(erase_kept)}/{len(sentences)} kept")
            
            if len(single_kept) > 0:
                print(f"  Reduction: {1-len(erase_kept)/len(single_kept):.0%}")
            
            single_ans = answer_question(llm, single_kept, query)
            erase_ans = answer_question(llm, erase_kept, query)
            
            print(f"\n  Single: {single_ans[:60]}... {'✅' if expected in single_ans else '❌'}")
            print(f"  ERASE:  {erase_ans[:60]}... {'✅' if expected in erase_ans else '❌'}")
            
            # Show what was kept
            print(f"\n  [Single kept ({len(single_kept)})]")
            for s in single_kept[:3]:
                print(f"    • {s[:70]}...")
            
            print(f"\n  [ERASE kept ({len(erase_kept)})]")
            for s in erase_kept[:3]:
                print(f"    • {s[:70]}...")
            print()
    
    asyncio.run(run())
    print("=" * 70)


if __name__ == "__main__":
    main()
