import json
import uuid

from redis import asyncio as aredis
from fastapi import FastAPI, Body


redis_client = aredis.from_url("redis://redis:6379", decode_responses=True)

app = FastAPI()

# [1] 클라이언트에서 질문(question)을 요청한다.
@app.post("/chats")
async def chat_handler(
    question: str = Body(..., embed=True),
):
    # [2] 결과 채널을 구독
    job_id = str(uuid.uuid4()) # 작업을 식별할 수 있는 랜덤 식별자 발금
    channel = f"result:{job_id}"

    pubsub = redis_client.pubsub()
    await pubsub.subscribe(channel)

    # [3] 답변 생성 작업 Enqueue
    job = {"id": job_id, "question": question}
    await redis_client.lpush("inference_queue", json.dumps(job))

    # [4] 답변 생성 결과를 돌려받기
    result = None
    async for message in pubsub.listen():
        if message["type"] == "message":
            result = message["data"]
            break
    return {"result": result}
