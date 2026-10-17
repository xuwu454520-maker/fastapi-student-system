"""AI 学生信息提取功能的测试。

分三层：
1. 单元测试：JSON 解析、字段归一化
2. 接口测试：FastAPI TestClient + 模拟模型响应（不调用真实 API、不花钱）
3. 真实调用：需要真实 API Key，另行手动验证
"""

import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import ai_extract
from ai_extract import _clean_str, _normalize, _parse_json, _to_optional_int
from main import app


client = TestClient(app)


# ==========================
# 模拟大模型返回（结构模仿 openai SDK）
# ==========================

class _Message:

    def __init__(self, content):
        self.content = content


class _Choice:

    def __init__(self, content):
        self.message = _Message(content)


class _Response:

    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:

    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error

    def create(self, **kwargs):
        if self.error:
            raise self.error
        return _Response(self.content)


class _Chat:

    def __init__(self, content=None, error=None):
        self.completions = _Completions(content, error)


class FakeClient:

    def __init__(self, content=None, error=None):
        self.chat = _Chat(content, error)


def _use_fake_model(monkeypatch, content=None, error=None):
    monkeypatch.setattr(
        ai_extract,
        "get_client",
        lambda: FakeClient(content, error)
    )


# ==========================
# 第一层：单元测试
# ==========================

@pytest.mark.parametrize("raw,expected", [
    (20, 20),
    ("20", 20),
    ("20岁", 20),
    (20.0, 20),
])
def test_to_optional_int_ok(raw, expected):
    assert _to_optional_int(raw) == expected


def test_to_optional_int_bad():
    assert _to_optional_int(None) is None
    assert _to_optional_int("abc") is None


def test_clean_str():
    assert _clean_str(" 张三 ") == "张三"
    assert _clean_str("") is None
    assert _clean_str(None) is None


def test_parse_json_plain():
    assert _parse_json('{"name": "张三"}') == {"name": "张三"}


def test_parse_json_code_fence():
    content = '```json\n{"name": "张三"}\n```'
    assert _parse_json(content) == {"name": "张三"}


def test_parse_json_with_extra_text():
    content = '好的，结果如下：{"name": "张三"} 希望有帮助'
    assert _parse_json(content) == {"name": "张三"}


def test_parse_json_invalid():
    with pytest.raises((json.JSONDecodeError, ValueError)):
        _parse_json("这不是 JSON")


def test_normalize_missing_fields():
    info = _normalize({"name": "张三"})
    assert info.name == "张三"
    assert info.age is None
    assert info.major is None


# ==========================
# 第二层：接口测试（模拟模型）
# ==========================

def test_extract_normal(monkeypatch):
    _use_fake_model(
        monkeypatch,
        '{"name": "张三", "age": 20, "major": "人工智能"}'
    )

    resp = client.post(
        "/ai/extract-student",
        json={"text": "张三，20岁，人工智能专业"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"name": "张三", "age": 20, "major": "人工智能"}
    # 验收标准：age 必须是整数
    assert isinstance(body["age"], int)


def test_extract_missing_age(monkeypatch):
    _use_fake_model(
        monkeypatch,
        '{"name": "张三", "age": null, "major": "人工智能"}'
    )

    resp = client.post(
        "/ai/extract-student",
        json={"text": "张三，人工智能专业"}
    )

    assert resp.status_code == 200
    # 验收标准：缺失信息不能被编造
    assert resp.json()["age"] is None


def test_extract_irrelevant_text(monkeypatch):
    _use_fake_model(
        monkeypatch,
        '{"name": null, "age": null, "major": null}'
    )

    resp = client.post(
        "/ai/extract-student",
        json={"text": "今天天气不错"}
    )

    assert resp.status_code == 200
    assert resp.json() == {"name": None, "age": None, "major": None}


def test_extract_blank_text(monkeypatch):
    # 全空格应在调用模型之前就被拦下
    monkeypatch.setattr(
        ai_extract,
        "get_client",
        lambda: pytest.fail("空输入不应该调用模型")
    )

    resp = client.post(
        "/ai/extract-student",
        json={"text": "   "}
    )

    assert resp.status_code == 400
    assert "不能为空" in resp.json()["detail"]


def test_extract_empty_string(monkeypatch):
    # 空字符串由 Pydantic 的 min_length 拦下
    resp = client.post(
        "/ai/extract-student",
        json={"text": ""}
    )

    assert resp.status_code == 422


def test_extract_llm_failure(monkeypatch):
    _use_fake_model(monkeypatch, error=RuntimeError("网络超时"))

    resp = client.post(
        "/ai/extract-student",
        json={"text": "张三，20岁，人工智能专业"}
    )

    assert resp.status_code == 502
    # 验收标准：不泄露 API Key / 内部细节
    assert "sk-" not in resp.text
    assert "网络超时" not in resp.text


def test_extract_invalid_json_from_model(monkeypatch):
    _use_fake_model(monkeypatch, "抱歉，我无法提取")

    resp = client.post(
        "/ai/extract-student",
        json={"text": "张三，20岁，人工智能专业"}
    )

    assert resp.status_code == 502
    assert "JSON" in resp.json()["detail"]


def test_extract_empty_choices(monkeypatch):
    """模型返回空 choices 时，不能崩成 500。"""

    class _EmptyResponse:
        choices = []

    class _EmptyCompletions:
        def create(self, **kwargs):
            return _EmptyResponse()

    class _EmptyChat:
        completions = _EmptyCompletions()

    class EmptyChoicesClient:
        chat = _EmptyChat()

    monkeypatch.setattr(ai_extract, "get_client", lambda: EmptyChoicesClient())

    resp = client.post(
        "/ai/extract-student",
        json={"text": "张三，20岁，人工智能专业"}
    )

    assert resp.status_code == 502


def test_extract_without_api_key(monkeypatch):
    def raise_no_key():
        raise HTTPException(
            status_code=500,
            detail="未配置 DEEPSEEK_API_KEY，请先在 .env 文件中填写"
        )

    monkeypatch.setattr(ai_extract, "get_client", raise_no_key)

    resp = client.post(
        "/ai/extract-student",
        json={"text": "张三，20岁，人工智能专业"}
    )

    assert resp.status_code == 500
    assert "未配置" in resp.json()["detail"]
