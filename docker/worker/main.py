## Llama 추론 프로그램
import redis


# redis = 엄청 빠른 초소형 데이터베이스(key:value)
redis_client= redis.from_url("redis://redis:6379", decode_response=True)


def run():
    # while True:
    pass

# python main.py를 직접 실행되었을 때만, run()을 호출
if __name__ == "__main__":
    run()