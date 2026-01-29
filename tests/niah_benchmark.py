"""
Needle in a Haystack (NIAH) style benchmark for ERASE.

Concept:
1. Create a "haystack" of distractor sentences
2. Insert a "needle" (target fact) at random position
3. Ask a query that requires the needle
4. Compare: Traditional RAG vs ERASE
5. Measure: Does the LLM find the needle with each approach?
"""
import random
from erase import ERASE, scope
from erase.schemas import MemoryChunk
from langchain_openai import ChatOpenAI


# Haystack distractors (various topics, all important-sounding)
HAYSTACK_SENTENCES = [
    "2023년 1분기 글로벌 시장 점유율은 23.5%로 전년 대비 2.1% 상승했습니다.",
    "신규 물류센터는 경기도 평택에 건설 예정이며 2024년 완공 목표입니다.",
    "클라우드 마이그레이션 프로젝트는 3단계로 진행되며 총 18개월이 소요됩니다.",
    "인사팀 조직개편으로 채용팀과 교육팀이 통합되었습니다.",
    "연간 R&D 투자 비중은 매출의 8.7%로 업계 평균을 상회합니다.",
    "고객만족도 조사 결과 NPS 점수가 72점으로 역대 최고치를 기록했습니다.",
    "ISO 27001 인증 갱신이 완료되어 정보보안 체계가 재확인되었습니다.",
    "신제품 베타 테스트 참여자는 5,000명이며 피드백 수집 중입니다.",
    "해외 지사 확장 계획에 따라 싱가포르 법인 설립을 추진 중입니다.",
    "에너지 효율화 프로젝트로 전년 대비 탄소 배출량 15% 감축을 달성했습니다.",
    "협력업체 평가 시스템이 개편되어 ESG 항목이 추가되었습니다.",
    "사내 교육 플랫폼 이용률이 78%로 목표치를 초과 달성했습니다.",
    "분기별 실적 발표는 매 분기 마지막 주 목요일에 진행됩니다.",
    "직원 복지 프로그램에 심리상담 서비스가 새롭게 포함되었습니다.",
    "재고관리 시스템 업그레이드로 재고 회전율이 12% 개선되었습니다.",
]

# Needles (specific facts to find)
NEEDLE_TEMPLATES = [
    {
        "needle": "김민수 부장의 비밀 프로젝트 코드명은 'AURORA'이며 예산은 35억원입니다.",
        "query": "김민수 부장의 비밀 프로젝트 코드명과 예산이 뭐야?",
        "expected_keywords": ["AURORA", "35억"],
    },
    {
        "needle": "다음 주 화요일 오후 3시에 대표이사와 긴급 회의가 예정되어 있습니다.",
        "query": "대표이사와의 회의가 언제야?",
        "expected_keywords": ["화요일", "3시"],
    },
    {
        "needle": "신규 AI 모델 'NEXUS-7'의 정확도는 94.3%이며 다음 달 출시 예정입니다.",
        "query": "새로운 AI 모델 정확도가 얼마야?",
        "expected_keywords": ["94.3%", "NEXUS"],
    },
]


class SingleScoredMemory:
    """Traditional RAG baseline."""
    
    def __init__(self, config):
        self._erase = ERASE(config)
        self._threshold = config.threshold.retention
    
    def retrieve(self, text: str, query: str) -> list[MemoryChunk]:
        all_chunks = self._erase.score_all(text, query=query)
        return [c for c in all_chunks if c.retention_score >= self._threshold]


def create_haystack(needle: str, haystack_size: int = 10) -> str:
    """Create a haystack with needle inserted at random position."""
    distractors = random.sample(HAYSTACK_SENTENCES, min(haystack_size, len(HAYSTACK_SENTENCES)))
    insert_pos = random.randint(0, len(distractors))
    distractors.insert(insert_pos, needle)
    return "\n".join(distractors)


def check_needle_found(chunks: list[MemoryChunk], keywords: list[str]) -> bool:
    """Check if needle keywords are in the retrieved chunks."""
    combined = " ".join(c.content for c in chunks)
    return all(kw in combined for kw in keywords)


@scope
def main(config):
    """Run NIAH benchmark."""
    erase = ERASE(config)
    single = SingleScoredMemory(config)
    llm = ChatOpenAI(model=config.model)
    
    print("=" * 70)
    print("Needle in a Haystack (NIAH) Benchmark for ERASE")
    print("=" * 70)
    print(f"Haystack size: 10 distractor sentences + 1 needle")
    print()
    
    results = []
    
    for i, template in enumerate(NEEDLE_TEMPLATES, 1):
        needle = template["needle"]
        query = template["query"]
        keywords = template["expected_keywords"]
        
        # Create haystack with needle
        haystack = create_haystack(needle, haystack_size=10)
        
        print(f"[Test {i}] Query: {query}")
        print(f"         Needle: {needle[:50]}...")
        print()
        
        # Traditional RAG
        single_chunks = single.retrieve(haystack, query)
        single_found = check_needle_found(single_chunks, keywords)
        
        # ERASE
        erase_chunks = erase(haystack, query)
        erase_found = check_needle_found(erase_chunks, keywords)
        
        # Compare chunk counts
        print(f"  [Traditional RAG]")
        print(f"    Chunks retrieved: {len(single_chunks)}")
        print(f"    Needle found: {'✅' if single_found else '❌'}")
        
        print(f"  [ERASE]")
        print(f"    Chunks retrieved: {len(erase_chunks)}")
        print(f"    Needle found: {'✅' if erase_found else '❌'}")
        
        # Noise ratio (lower is better for ERASE)
        noise_reduction = 1 - (len(erase_chunks)/len(single_chunks)) if single_chunks else 0
        print(f"  Noise reduction: {noise_reduction:.0%}")
        
        # Ask LLM to answer using each context
        if erase_chunks:
            erase_context = "\n".join(c.content for c in erase_chunks)
            llm_prompt = f"Based on this context, answer: {query}\n\nContext:\n{erase_context}"
            llm_answer = llm.invoke(llm_prompt).content
            llm_correct = all(kw in llm_answer for kw in keywords)
            print(f"  LLM answer (ERASE context): {llm_answer[:80]}...")
            print(f"  LLM correct: {'✅' if llm_correct else '❌'}")
        
        print()
        results.append({
            "single_found": single_found,
            "erase_found": erase_found,
            "single_chunks": len(single_chunks),
            "erase_chunks": len(erase_chunks),
        })
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    single_success = sum(1 for r in results if r["single_found"])
    erase_success = sum(1 for r in results if r["erase_found"])
    avg_single = sum(r["single_chunks"] for r in results) / len(results)
    avg_erase = sum(r["erase_chunks"] for r in results) / len(results)
    
    print(f"  Traditional RAG: {single_success}/{len(results)} needles found, avg {avg_single:.1f} chunks")
    print(f"  ERASE: {erase_success}/{len(results)} needles found, avg {avg_erase:.1f} chunks")
    print(f"  Context reduction: {1 - (avg_erase/avg_single):.0%} less noise")
    print("=" * 70)


if __name__ == "__main__":
    main()
