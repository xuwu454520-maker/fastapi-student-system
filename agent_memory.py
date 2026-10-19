import json
import os

import redis
from dotenv import load_dotenv

load_dotenv()

redis_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6380/0"),
    decode_responses=True,
    socket_connect_timeout=3,
    socket_timeout=3,
)

MEMORY_TTL_SECONDS = 60 * 60 * 24


def get_history(session_id: str) -> list[dict]:
    key = f"agent:memory:{session_id}"
    raw = redis_client.get(key)

    if not raw:
        return []

    try:
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def save_history(
    session_id: str,
    history: list[dict],
) -> None:
    key = f"agent:memory:{session_id}"

    redis_client.set(
        key,
        json.dumps(history[-10:], ensure_ascii=False),
        ex=MEMORY_TTL_SECONDS,
    )


def clear_history(session_id: str) -> None:
    redis_client.delete(f"agent:memory:{session_id}")


def check_redis() -> bool:
    return bool(redis_client.ping())


def set_session_owner(session_id: str, username: str) -> None:
    """记录会话所属用户，并设置 24 小时过期时间。"""
    key = f"agent:owner:{session_id}"
    redis_client.set(
        key,
        username,
        ex=MEMORY_TTL_SECONDS,
    )


def get_session_owner(session_id: str) -> str | None:
    """获取会话所属用户。"""
    return redis_client.get(f"agent:owner:{session_id}")


def delete_session(session_id: str) -> None:
    """同时删除会话记忆和所有者记录。"""
    redis_client.delete(
        f"agent:memory:{session_id}",
        f"agent:owner:{session_id}",
    )
