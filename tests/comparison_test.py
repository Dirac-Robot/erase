"""
Comparison: Single-scored Memory (Traditional RAG) vs Dual-scored Memory (ERASE)

Key insight: ERASE can exclude "important but off-topic" information
that traditional RAG would include (because it only looks at relevance).
"""
from erase import ERASE, scope
from erase.schemas import MemoryChunk


class SingleScoredMemory:
    """Traditional RAG: Only uses relevance score, no exclusion mechanism."""
    
    def __init__(self, config):
        self._erase = ERASE(config)
        self._threshold = config.threshold.retention
    
    def retrieve(self, text: str, query: str) -> list[MemoryChunk]:
        all_chunks = self._erase.score_all(text, query=query)
        return [c for c in all_chunks if c.retention_score >= self._threshold]


@scope
def main(config):
    # 핵심 테스트 케이스: "중요하지만 지금은 off-topic"인 정보 포함
    memory_bank = """
    [프로젝트 A 미팅 - 어제]
    - 프로젝트 A 예산: 10억원 확정
    - 프로젝트 A 담당자: 김철수 팀장
    - 프로젝트 A 마감일: 6월 30일
    
    [프로젝트 B 미팅 - 오늘]
    - 프로젝트 B 예산: 5억원 신청 중
    - 프로젝트 B 담당자: 박영희 대리
    - 프로젝트 B 마감일: 9월 15일
    - 프로젝트 B 기술 스택: Python, FastAPI
    
    [잡담]
    - 회의실 에어컨이 고장나서 수리 요청함
    - 커피머신 캡슐 재주문 필요
    """
    
    # 프로젝트 B에 대해서만 질문
    query = "프로젝트 B 예산이랑 마감일이 언제야?"
    
    print("=" * 70)
    print("Single-scored vs Dual-scored Memory Comparison")
    print("=" * 70)
    print(f"Query: '{query}'")
    print(f"Retention threshold: {config.threshold.retention}")
    print(f"Erasure threshold: {config.threshold.erasure}")
    print()
    
    # Get all chunks with scores first
    erase = ERASE(config)
    all_chunks = erase.score_all(memory_bank, query)
    
    print("[All chunks with scores]")
    print("-" * 70)
    for c in all_chunks:
        print(f"  R={c.retention_score:.2f} E={c.erasure_score:.2f} | {c.content[:50]}...")
    print()
    
    # Single-scored (Traditional RAG)
    single = SingleScoredMemory(config)
    single_results = single.retrieve(memory_bank, query)
    
    print("[Single-scored Memory (Traditional RAG)]")
    print("-" * 70)
    print(f"Retrieved {len(single_results)} chunks (retention >= {config.threshold.retention}):")
    for c in single_results:
        print(f"  R={c.retention_score:.2f} | {c.content[:55]}...")
    print()
    
    # Dual-scored (ERASE)
    dual_results = erase(memory_bank, query)
    
    print("[Dual-scored Memory (ERASE)]")
    print("-" * 70)
    print(f"Retrieved {len(dual_results)} chunks (R >= {config.threshold.retention} AND E < {config.threshold.erasure}):")
    for c in dual_results:
        print(f"  R={c.retention_score:.2f} E={c.erasure_score:.2f} | {c.content[:50]}...")
    print()
    
    # 차이점 분석
    print("[🔍 Key Difference: What Traditional RAG keeps but ERASE excludes]")
    print("-" * 70)
    
    excluded_by_erase = []
    for c in all_chunks:
        in_single = c.retention_score >= config.threshold.retention
        in_dual = c.retention_score >= config.threshold.retention and c.erasure_score < config.threshold.erasure
        if in_single and not in_dual:
            excluded_by_erase.append(c)
    
    if excluded_by_erase:
        for c in excluded_by_erase:
            print(f"  ⚠️ R={c.retention_score:.2f} E={c.erasure_score:.2f} | {c.content[:50]}...")
        print()
        print("→ Traditional RAG: 'retention 높으니까 가져와!'")
        print("→ ERASE: '중요하긴 한데 프로젝트 B 질문에는 방해되니까 배제!'")
    else:
        print("No difference in this case (LLM may not have assigned high erasure)")
    
    print()
    print("=" * 70)
    print("Summary:")
    print(f"  Single-scored: {len(single_results)} chunks (may include off-topic noise)")
    print(f"  Dual-scored:   {len(dual_results)} chunks (focused on query)")
    print(f"  Excluded by ERASE: {len(excluded_by_erase)} chunks")
    print("=" * 70)


if __name__ == "__main__":
    main()
