# SQLAlchemy 사용에 필요한 기본 설정
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# DB 접속 정보(DB 종류, 주소, 포트번호, 사용자, 비밀번호, DB 이름)
# sqlite:// -> sqlite를 사용하겠다
# /./test.db -> 현재 프로젝트 경로에 test.db라는 이름의 파일을 만들어라
DATABASE_URL = "sqlite:///./test.db"

# Engine: DB와 연결을 제공하는 객체
engine = create_engine(DATABASE_URL)

# Session: DB 작업단위

# sessionmaker: 클래스를 만들어주는 함수
# sessionmaker()의 역할 -> class SessionFactory(...): 같이 클래스 만들어 줌


# SesstionFactory: 세션을 생성하는 클래스
SessionFactory = sessionmaker(
    bind=engine, # 엔진을 연결

    # 기본 옵션
    autocommit=False, # 자동으로 commit() 실행
    autoflush=False,  # 자동으로 flush() 실행
    expire_on_commit=False, # commit() 후에 자동으로 데이터 만료(expire)
    )

# return - > 호출부 값 반환 & 실행한 함수 종료
# yield -> 호출부로 값 반환 & 실행한 함수 일시정지(종료X)
def get_session():
    session = SessionFactory()
    
    try:
        yield session # 일시정지
    finally:
        session.close()