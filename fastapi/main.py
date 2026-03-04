import anyio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Path, Query, Body, status, HTTPException, Depends, BackgroundTasks
from sqlalchemy import select

from db_connection import SessionFactory, get_session
from db_connection_async import get_async_session
from models import User
from schema import UserSignUpRequest, UserResponse, UserUpdateRequest


def send_email(name: str):
    import time
    time.sleep(5) # 5초 대기
    print(f"{name}에게 이메일 전송이 완료되었습니다.")


# 스레드 풀 개수
@asynccontextmanager
# lifespan -> FastAPI 서버가 실행되고 종료될 때, 특정 리소스를 생성하고 정리하는 기능
async def lifespan(_):
    # 서버 실행될 때, 실행되는 부분
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = 200 # 스레들 풀 개수를 200개로 증량
    yield
    # 서버 종료될 때, 실행되는 부분

app =FastAPI(lifespan=lifespan)

# # 서버에 GET /hello 요청이 들어오면, hello_handler를 실행한다
# @app.get("/hello")
# def hello_handler():
#     return {"ping": "pong"}




# 전체 사용자 조회 API
@app.get(
        "/users",
        status_code=status.HTTP_200_OK,
        response_model=list[UserResponse],
)
async def get_users_handler(

    # 요청이 시작 -> session이 생성
    # 응답 반환 -> session.close()
    session = Depends(get_async_session),
):
    # statment(구문) -> SELECT * FROM user
    stmt = select(User)
    result = await session.execute(stmt)

    # [user1, user2, ...]
    users = result.scalars().all()
    return users

# 회원 검색 API
# HTTP Method: GET, POST, PUT, PATCH, DELETE
# Query Parameter ->  ?key=value 형태로 Path 뒤에 붙는 값
# 데이터 조회시 부가 조건을 명시(필터링, 정렬, 검색 ,페이지네이션 등)
@app.get(
        "/users/search",
        status_code=status.HTTP_200_OK,
        response_model=UserResponse,
)
def search_user_handler(
    # name이라는 key로 넘어오는 Query Parameter 값을 사용하겠다
    name: str = Query(..., min_length=2),  # ... -> 필수값(required)
    age: int = Query(None, ge=1), # default 값 지정 -> 선택적(optional)
):
    return {"id":0,"name": name, "age": age}


# {user_id}번/단일 사용자 조회 API
# Path(경로) + Parameter(매개변수) -> 동적으로 바뀌는 값을 한 번에 처리
# Path Parameter에 type hint 추가하면 -> 명시한 타입에 맞는지 검사  & 보장

# GET/users/1?field=id -> id 반환
# GET/users/1?field=name -> name 반환
# GET/users/1 (없으면)-> id,name 반환
@app.get(
        "/users/{user_id}", 
        status_code=status.HTTP_200_OK,
        response_model=UserResponse,
)
async def get_user_handler(
    user_id: int = Path(..., ge=1, description="사용자의 ID"),
    session = Depends(get_async_session),
):
    
    stmt = select(User).where(User.id == user_id)
    result = await session.execute(stmt)
    user = result.scalar() # 1개의 데이터를 가져올 때

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="존재하지 않는 사용자의 ID입니다.",
        )
    return user


    # gt : 초과
    # ge : 이상
    # lt : 미만
    # le : 이하
    # max_digits: 최대 자리수 000000


# 회원가입 API
@app.post(
        "/users/sign-up", 
        status_code=status.HTTP_201_CREATED,

        # 응답은 UserSignUpResponse 데이터 구조를 따라야한다
        # UserSignUpResponse(id:int, name:str, age: int | None)

        # 1) 서버에서 원하는 데이터 형식으로 응답이 반환되는지 검증
        # 2) 노출되면 안 되는 값을 자동으로 제거
        # 3) API 문서에 예상되는 응답 출력
        response_model=UserResponse,
)
async def sign_up_handler(
    body: UserSignUpRequest,
    background_tasks: BackgroundTasks,  # BackgroundTasks 객체를 주입
    session = Depends(get_async_session),
):
    # 함수에 선언한 매개변수의 타입힌트가 BaseModel을 상속 받은 경우, 요청 본문에서 가져옴
    # 데이터 가져오면서, 타입힌트에 선언한 데이터 구조와 맞는지 검사

    # body = UserSignUpRequest(name=..., age=...)
    # body 데이터가 문제 없으면 -> 핸들러 함수로 전달
    # body 데이터가 문제 있으면 -> 즉시  실행이 멈추고, 422 에러
    
    # SQLAlchemy ORM을 통해 새로운  user 인스턴스 생성
    # id -> 데이터베이스가 관리하도록 위임
    new_user = User(name=body.name, age=body.age)

    session.add(new_user)
    await session.commit() # DB에 저장

    # 이메일 전송 작업 BT(background_tasks)에 등록
    background_tasks.add_task(send_email, body.name)
    return new_user


# 사용자 정보 수정 API
# PUT :전체업데이트-> {name, age} 한 번에 교체
# PATCH : 일부분 업데이트 -> name | age 하나씩 교체
@app.patch(
        "/users/{user_id}",
        status_code=status.HTTP_200_OK,
        response_model=UserResponse,
)
async def update_user_handler(
    user_id: int = Path(..., ge=1),
    body: UserUpdateRequest = Body(...),
    session = Depends(get_async_session),
):

    # 1) body 데이터 검증
    if body.name is None and body.age is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="수정할 데이터가 없습니다.",
        )

    # 2) 사용자 조회
    
    stmt = select(User).where(User.id == user_id)
    result = await session.execute(stmt)
    user = result.scalar() 

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="존재하지 않는 사용자의 ID입니다.",
)

    if body.name is not None:
        user.name = body.name

    if body.age is not None:
        user.age = body.age

    await session.commit()

    # 3) 응답 반환
    return user


# 사용자 삭제(회원탈퇴) API
@app.delete(
        "/users/{user_id}",
        status_code=status.HTTP_204_NO_CONTENT,  # 응답 본문 생성하지 않음
)
async def delete_user_handler(
    user_id: int = Path(..., ge=1),
    session = Depends(get_async_session),
):

    stmt = select(User).where(User.id == user_id)
    result = await session.execute(stmt)
    user = result.scalar()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="존재하지 않는 사용자의 ID입니다.",
        )
        
    session.delete(user)
    await session.commit()





### 실습

# GET /Items/{item_name}
# item_name: str & 최대 글자수(max_length) 6
# 응답 형식: {"item_name": ...}

@app.get("/items/{item_name}")
def get_item_handler(
    item_name: str = Path(..., max_length=6),
):
    return {"item_name": item_name}
