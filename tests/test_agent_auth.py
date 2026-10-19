from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_agent_chat_requires_login():
    """未登录用户不能调用 Agent。"""
    response = client.post(
        "/agent/chat",
        json={
            "session_id": "test-session-001",
            "message": "你好",
        },
    )

    assert response.status_code == 401


def test_agent_memory_requires_login():
    """未登录用户不能读取 Agent 记忆。"""
    response = client.get(
        "/agent/memory/test-session-001",
    )

    assert response.status_code == 401


def test_agent_delete_memory_requires_login():
    """未登录用户不能删除 Agent 记忆。"""
    response = client.delete(
        "/agent/memory/test-session-001",
    )

    assert response.status_code == 401


# =========================================
# 双用户隔离测试
# =========================================

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from main import app
from routers.auth import get_current_user
import routers.agent as agent_router_module


@pytest.fixture
def isolated_agent(monkeypatch):
    """模拟两个用户和 Redis 会话，避免操作真实会话数据。"""
    current = {"username": "user_a"}
    owners = {}
    histories = {}

    def fake_set_owner(session_id, username):
        owners[session_id] = username

    def fake_get_owner(session_id):
        return owners.get(session_id)

    def fake_get_history(session_id):
        return histories.get(session_id, [])

    def fake_delete_session(session_id):
        owners.pop(session_id, None)
        histories.pop(session_id, None)

    monkeypatch.setattr(
        agent_router_module, "set_session_owner", fake_set_owner
    )
    monkeypatch.setattr(
        agent_router_module, "get_session_owner", fake_get_owner
    )
    monkeypatch.setattr(
        agent_router_module, "get_history", fake_get_history
    )
    monkeypatch.setattr(
        agent_router_module, "delete_session", fake_delete_session
    )

    app.dependency_overrides[get_current_user] = lambda: (
        SimpleNamespace(username=current["username"])
    )

    with TestClient(app) as test_client:
        yield test_client, current, owners, histories

    app.dependency_overrides.pop(get_current_user, None)


def test_users_cannot_read_each_others_memory(isolated_agent):
    client, current, owners, histories = isolated_agent

    # 用户 A 创建会话
    response = client.post("/agent/session")
    assert response.status_code == 200

    session_id = response.json()["session_id"]
    assert owners[session_id] == "user_a"

    # 准备模拟聊天记录
    histories[session_id] = [
        {"role": "user", "content": "用户 A 的测试内容"}
    ]

    # 用户 B 尝试读取用户 A 的记忆
    current["username"] = "user_b"
    response = client.get(f"/agent/memory/{session_id}")

    assert response.status_code == 404


def test_users_cannot_delete_each_others_memory(isolated_agent):
    client, current, owners, histories = isolated_agent

    # 用户 A 创建会话
    response = client.post("/agent/session")
    assert response.status_code == 200

    session_id = response.json()["session_id"]
    histories[session_id] = [
        {"role": "user", "content": "保留这条测试记录"}
    ]

    # 用户 B 尝试删除用户 A 的会话
    current["username"] = "user_b"
    response = client.delete(f"/agent/memory/{session_id}")

    assert response.status_code == 404
    assert session_id in owners
    assert session_id in histories


def test_owner_can_read_and_delete_own_memory(isolated_agent):
    client, current, owners, histories = isolated_agent

    # 用户 A 创建自己的会话
    response = client.post("/agent/session")
    assert response.status_code == 200

    session_id = response.json()["session_id"]
    histories[session_id] = [
        {"role": "user", "content": "我的测试记录"}
    ]

    # 用户 A 可以读取自己的记录
    response = client.get(f"/agent/memory/{session_id}")
    assert response.status_code == 200
    assert response.json()["history"][0]["content"] == "我的测试记录"

    # 用户 A 可以删除自己的记录
    response = client.delete(f"/agent/memory/{session_id}")
    assert response.status_code == 200
    assert session_id not in owners
    assert session_id not in histories
