import os

import redis


# 同理，从环境变量读取，默认值对应你本地的 Redis
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6380")),
    decode_responses=True
)


def get_redis():

    return redis_client