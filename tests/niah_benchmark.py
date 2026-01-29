"""
NIAH Benchmark for ERASE - Fair Comparison Version

Features:
- Separate single-scored prompt (fair baseline)
- Async parallel LLM calls
- Configurable haystack size and chunk size
"""
import asyncio
import random
from erase import ERASE, scope
from erase.schemas import MemoryChunk
from erase.single_scored import SingleScoredMemory


# Extended haystack distractors
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
    "신규 파트너십으로 유럽 시장 진출 기반이 마련되었습니다.",
    "모바일 앱 다운로드 수가 500만 건을 돌파했습니다.",
    "품질관리 프로세스 개선으로 불량률이 0.3% 이하로 감소했습니다.",
    "데이터센터 확장 공사가 3분기 중 완료될 예정입니다.",
    "고객센터 응대 시간이 평균 2분 30초로 단축되었습니다.",
    "신규 채용 공고에 1만 명 이상의 지원자가 접수되었습니다.",
    "특허 출원 건수가 전년 대비 35% 증가했습니다.",
    "온라인 판매 비중이 전체 매출의 45%를 차지하게 되었습니다.",
    "생산라인 자동화율이 85%에 도달했습니다.",
    "고객 리텐션율이 92%로 업계 최고 수준을 유지하고 있습니다.",
    "신규 서비스 론칭 후 3개월 만에 10만 사용자를 확보했습니다.",
    "공급망 다변화로 원자재 조달 리스크가 크게 감소했습니다.",
    "사내 스타트업 프로그램에서 5개 팀이 최종 선발되었습니다.",
    "연구소 신축 건물이 내년 상반기 준공 예정입니다.",
    "글로벌 브랜드 인지도 조사에서 상위 10위권에 진입했습니다.",
]

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
]


def create_haystack(needle: str, haystack_size: int = 10) -> str:
    """Create a haystack with needle inserted at random position."""
    if haystack_size <= len(HAYSTACK_SENTENCES):
        distractors = random.sample(HAYSTACK_SENTENCES, haystack_size)
    else:
        distractors = random.choices(HAYSTACK_SENTENCES, k=haystack_size)
    
    insert_pos = random.randint(0, len(distractors))
    distractors.insert(insert_pos, needle)
    return "\n".join(distractors)


def check_needle_found(chunks: list[MemoryChunk], keywords: list[str]) -> bool:
    """Check if needle keywords are in the retrieved chunks."""
    combined = " ".join(c.content for c in chunks)
    return all(kw in combined for kw in keywords)


async def run_test_async(erase: ERASE, single: SingleScoredMemory, template: dict,
                         haystack_size: int) -> dict:
    """Run a single test asynchronously."""
    needle = template["needle"]
    query = template["query"]
    keywords = template["expected_keywords"]
    
    haystack = create_haystack(needle, haystack_size=haystack_size)
    
    loop = asyncio.get_event_loop()
    single_task = loop.run_in_executor(None, single, haystack, query)
    erase_task = loop.run_in_executor(None, erase, haystack, query)
    
    single_chunks, erase_chunks = await asyncio.gather(single_task, erase_task)
    
    single_found = check_needle_found(single_chunks, keywords)
    erase_found = check_needle_found(erase_chunks, keywords)
    
    return {
        "single_found": single_found,
        "erase_found": erase_found,
        "single_chunks": len(single_chunks),
        "erase_chunks": len(erase_chunks),
        "noise_reduction": 1-(len(erase_chunks)/len(single_chunks)) if single_chunks else 0,
    }


@scope
def main(config):
    """Run NIAH benchmark with fair comparison."""
    erase = ERASE(config)
    single = SingleScoredMemory(config)  # Uses separate single-score prompt
    
    print("=" * 70)
    print("NIAH Benchmark - Fair Comparison (Separate Prompts)")
    print("=" * 70)
    print("Single-scored: relevance-only prompt")
    print("ERASE: dual-scored prompt (retention + erasure)")
    print()
    
    configs = [10, 20, 30]  # Haystack sizes
    
    async def run_all():
        for haystack_size in configs:
            print(f"[Haystack: {haystack_size} sentences]", end=" ", flush=True)
            
            tasks = [
                run_test_async(erase, single, template, haystack_size)
                for template in NEEDLE_TEMPLATES
            ]
            results = await asyncio.gather(*tasks)
            
            avg_single = sum(r["single_chunks"] for r in results)/len(results)
            avg_erase = sum(r["erase_chunks"] for r in results)/len(results)
            avg_noise = sum(r["noise_reduction"] for r in results)/len(results)
            
            single_ok = sum(1 for r in results if r["single_found"])
            erase_ok = sum(1 for r in results if r["erase_found"])
            
            print(f"Single: {avg_single:.0f} ({single_ok}/2) | ERASE: {avg_erase:.0f} ({erase_ok}/2) | Noise↓ {avg_noise:.0%}")
    
    asyncio.run(run_all())
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
