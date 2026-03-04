# 비동기 SQLAlchemy 설정 파일
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession


DATABASE_URL = "sqlite+aiosqlite:///./test.db"

async_engine = create_async_engine(DATABASE_URL)

AsyncSessionFactory = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)

async def get_async_session():
    session = AsyncSessionFactory()
    try:
        yield session
    finally:
        # 세선을 종료하면서 네트워크를 사용하기 때문에 await
        yield session.close()