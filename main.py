from erase import ERASE, scope


@scope
def main(config):
    erase = ERASE(config)

    sample_text = """
    오늘 회의에서 김철수 팀장이 다음 분기 목표를 발표했습니다.
    매출 목표는 50억원이고, 신규 고객 100명 유치가 핵심 KPI입니다.
    회의실이 좀 추웠어요. 커피가 맛있었습니다.
    프로젝트 마감일은 3월 15일로 확정되었습니다.
    """

    print("=== ERASE Demo ===")
    print(f"Model: {config.model}")
    print(f"Retention threshold: {config.threshold.retention}")
    print(f"Erasure threshold: {config.threshold.erasure}")
    print()

    memories = erase(sample_text)

    print(f"Committed {len(memories)} memories:")
    for mem in memories:
        print(f"  [{mem.id}] R={mem.retention_score:.2f} E={mem.erasure_score:.2f}")
        print(f"       {mem.content[:50]}...")
        print()


if __name__ == "__main__":
    main()
