import re
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from routers.auth import get_current_user
from models.user import UserDB
from agent_memory import (
    check_redis,
    get_history,
    clear_history,
    set_session_owner,
    get_session_owner,
    delete_session,
)
from agent_service import AgentService


router = APIRouter(prefix="/agent", tags=["AI Agent"])

agent_service = AgentService()


class AgentChatRequest(BaseModel):
    session_id: str = Field(
        min_length=1,
        max_length=100,
        description="会话 ID",
    )
    message: str = Field(
        min_length=1,
        max_length=4000,
        description="用户的问题",
    )


def verify_session_owner(session_id: str, username: str):
    """只允许会话所属用户访问该会话。"""
    owner = get_session_owner(session_id)

    if owner is None or owner != username:
        raise HTTPException(
            status_code=404,
            detail="会话不存在或无权访问",
        )


@router.get("/health")
def agent_health():
    try:
        check_redis()
        return {"status": "ok", "redis": "connected"}
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Redis 连接失败，请检查 Redis 服务和 REDIS_URL",
        )


@router.post("/session")
def create_agent_session(
    current_user: UserDB = Depends(get_current_user),
):
    session_id = str(uuid.uuid4())

    set_session_owner(session_id, current_user.username)

    return {"session_id": session_id}


@router.post("/chat")
async def chat_with_agent(
    request: AgentChatRequest,
    current_user: UserDB = Depends(get_current_user),
):
    session_id = request.session_id.strip()
    message = request.message.strip()

    if not session_id or not message:
        raise HTTPException(
            status_code=422,
            detail="session_id 和 message 不能为空",
        )

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", session_id):
        raise HTTPException(
            status_code=422,
            detail="session_id 只能包含字母、数字、下划线和连字符",
        )

    verify_session_owner(session_id, current_user.username)

    try:
        return await agent_service.chat(session_id, message)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Agent 调用失败，请检查服务终端中的错误信息",
        )


@router.get("/memory/{session_id}")
def read_agent_memory(
    session_id: str,
    current_user: UserDB = Depends(get_current_user),
):
    verify_session_owner(session_id, current_user.username)

    return {
        "session_id": session_id,
        "history": get_history(session_id),
    }


@router.delete("/memory/{session_id}")
def delete_agent_memory(
    session_id: str,
    current_user: UserDB = Depends(get_current_user),
):
    verify_session_owner(session_id, current_user.username)

    delete_session(session_id)

    return {
        "session_id": session_id,
        "message": "会话记忆已清除",
    }
