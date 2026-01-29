from erase import ERASE, scope


@scope
def main(config):
    erase = ERASE(config)

    # 테스트 데이터: 여러 맥락이 섞인 회의록
    memory_bank = """
    [회의록 - 2024년 1월 종합]
    
    1. 마케팅팀: 1분기 예산 3억원, SNS 집중. 목표 신규고객 1만명.
    2. 개발팀: 신규 API 3월 출시 예정. 담당자 이민수.
    3. 인사팀: 김영희 대리 다음달 휴가. 신입 2명 채용 예정.
    4. 재무팀: 작년 매출 100억 달성. 올해 목표 150억.
    5. 법무팀: A사와 NDA 체결 완료. B사 소송 진행 중 (민감정보).
    """

    print("=== ERASE: Query-aware Scoring Demo ===")
    print(f"Retention threshold: {config.threshold.retention}")
    print(f"Erasure threshold: {config.threshold.erasure}")
    print()

    # 테스트 1: 쿼리 없이 (일반 요약)
    print("=" * 50)
    print("[TEST 1] No query (general summarization)")
    print("=" * 50)
    memories = erase(memory_bank)
    print(f"Committed {len(memories)} memories:")
    for mem in memories:
        print(f"  R={mem.retention_score:.2f} E={mem.erasure_score:.2f} | {mem.content[:50]}...")
    print()

    # 테스트 2: 마케팅 관련 쿼리
    print("=" * 50)
    print("[TEST 2] Query: '마케팅 예산과 목표가 뭐야?'")
    print("=" * 50)
    memories = erase(memory_bank, query="마케팅 예산과 목표가 뭐야?")
    print(f"Committed {len(memories)} memories:")
    for mem in memories:
        print(f"  R={mem.retention_score:.2f} E={mem.erasure_score:.2f} | {mem.content[:50]}...")
    print()

    # 테스트 3: 개발팀 관련 쿼리
    print("=" * 50)
    print("[TEST 3] Query: '개발팀 일정 알려줘'")
    print("=" * 50)
    memories = erase(memory_bank, query="개발팀 일정 알려줘")
    print(f"Committed {len(memories)} memories:")
    for mem in memories:
        print(f"  R={mem.retention_score:.2f} E={mem.erasure_score:.2f} | {mem.content[:50]}...")


if __name__ == "__main__":
    main()
