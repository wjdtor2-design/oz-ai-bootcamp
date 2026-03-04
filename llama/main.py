# LLM (Large Language Model)
# 입력 : 프롬프트(prompt) -> (LLM) -> 출력: 문장

# role
# system: 모델의 규칙, 행동방식, 성격
# user: 사용자들이 질문, 요청
# assistant: 이전에 모델이 생성한 답변

import asyncio
from contextlib import asynccontextmanager

from llama_cpp import Llama
from fastapi import FastAPI, Request, Body
from fastapi.responses import StreamingResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 실행 전, 준비되어야 하는 리소스
    app.state.llm = Llama(
        model_path="./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf", # 모델경로
        c_ctx=4096, # 컨텍스트 사이즈
        n_threads=2, # CPU 스레드
        verbose=False, # 로그 출력을 간단하게  /true시 상세하게
        chat_format="llama-3", # 응답 생성 포맷
    )
    yield
    # 서버 실행 후, 정리해야 하는 리소스


app = FastAPI(lifespan=lifespan)



# LLM에 대한 규칙
SYSTEM_PROMPT = (
    "You are a concise assistant. "
    "Always reply in the same language as the user's input. "
    "Do not change the language. "
    "Do not mix languages."
)

@app.post("/chats")
async def generate_chat_handler(
    request: Request,
    user_input: str = Body(..., embed=True),
):
    
    async def event_generator():
        llm = request.app.state.llm
        response = llm.create_chat_completion(
            messages=[
                {"role": "sysrem", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            max_tokens=256,
            temperature=0.7,
            stream=True,
        )

        for chunk in response:
            token = chunk["choices"][0]["delta"].get("content")
            if token:
                yield token
                await asyncio.sleep(0)
    
    return StreamingResponse(
        event_generator(),   # 제너레이터
        media_type="text/event-stream",
    )

