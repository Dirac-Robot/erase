"""Demo: Conversation Memory with ERASE filtering."""
from erase import ConversationMemory, scope


@scope
def main(config):
    memory = ConversationMemory(config)
    
    print("=" * 60)
    print("[ERASE Conversation Memory Demo]")
    print("=" * 60)
    print()
    
    # 여러 주제의 대화 시뮬레이션
    conversations = [
        ("user", "다음주에 제주도 휴가 가려고 하는데 추천해줘"),
        ("assistant", "제주도 동쪽 코스 추천드려요. 성산일출봉, 섭지코지, 우도가 좋아요."),
        ("user", "숙소는 어디가 좋을까?"),
        ("assistant", "서귀포 쪽 펜션이나 호텔 추천드려요. 중문 관광단지도 좋습니다."),
        ("user", "이번 분기 매출 목표 정리해줘"),
        ("assistant", "이번 분기 매출 목표는 50억원입니다. 주요 KPI는 신규 고객 1000명 유치입니다."),
        ("user", "경쟁사 분석 자료도 필요해"),
        ("assistant", "경쟁사 A사는 최근 신제품을 출시했고, B사는 가격 인하 전략을 쓰고 있어요."),
        ("user", "오늘 저녁 뭐 먹지?"),
        ("assistant", "근처에 새로 생긴 이탈리안 레스토랑 어때요? 파스타가 맛있대요."),
    ]
    
    # 대화 추가
    print("[Adding conversations to memory...]")
    for role, content in conversations:
        memory.add(role, content)
        print(f"  {role}: {content[:40]}...")
    print()
    
    # 테스트 1: 업무 관련 쿼리
    print("=" * 60)
    print("[Query 1] '매출 목표가 뭐였지?'")
    print("=" * 60)
    chunks = memory.retrieve("매출 목표가 뭐였지?")
    print(f"Retrieved {len(chunks)} relevant chunks:")
    for c in chunks:
        print(f"  R={c.retention_score:.2f} E={c.erasure_score:.2f} | {c.content[:50]}...")
    print()
    
    # 테스트 2: 휴가 관련 쿼리
    print("=" * 60)
    print("[Query 2] '제주도 여행 계획 다시 알려줘'")
    print("=" * 60)
    chunks = memory.retrieve("제주도 여행 계획 다시 알려줘")
    print(f"Retrieved {len(chunks)} relevant chunks:")
    for c in chunks:
        print(f"  R={c.retention_score:.2f} E={c.erasure_score:.2f} | {c.content[:50]}...")
    print()
    
    # 테스트 3: get_context 사용
    print("=" * 60)
    print("[Query 3] get_context('경쟁사 분석')")
    print("=" * 60)
    context = memory.get_context("경쟁사 분석", max_chars=500)
    print(f"Context (max 500 chars):\n{context}")


if __name__ == "__main__":
    main()
