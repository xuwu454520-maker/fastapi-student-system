import json
import logging
import re
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ai_chat import DEEPSEEK_MODEL, get_client


logger = logging.getLogger(__name__)

router = APIRouter(tags=["AI 学生信息提取"])


# ==========================
# 请求 / 响应模型
# ==========================

class ExtractRequest(BaseModel):

    text: str = Field(
        min_length=1,
        max_length=1000,
        description="包含学生信息的中文文本",
        examples=["张三，20岁，人工智能专业"]
    )


class StudentInfo(BaseModel):

    name: Optional[str] = Field(
        default=None,
        description="学生姓名，文本中未提及时为 null"
    )

    age: Optional[int] = Field(
        default=None,
        description="学生年龄（整数），文本中未提及时为 null"
    )

    major: Optional[str] = Field(
        default=None,
        description="学生专业，文本中未提及时为 null"
    )


# ==========================
# 提示词：要求模型只输出 JSON
# ==========================

SYSTEM_PROMPT = """你是一个严谨的学生信息提取器。
从用户提供的文本中提取学生信息，只输出一个 JSON 对象，不要输出任何解释、Markdown 或代码块。
JSON 格式固定为：
{"name": "姓名", "age": 整数, "major": "专业"}
规则：
1. 文本中没有提到的字段，其值必须是 null。
2. 绝对不能猜测或编造缺失的信息。
3. age 必须是 JSON 整数类型（例如 20），不能是字符串。
4. 只输出 JSON 本身。"""


# ==========================
# 辅助函数
# ==========================

def _clean_str(value) -> Optional[str]:
    """把值整理成字符串；空字符串或 None 都返回 None。"""
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _to_optional_int(value) -> Optional[int]:
    """尽量转成整数，兼容 "20"、"20岁" 这类返回。"""
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value))
        return int(match.group()) if match else None


def _parse_json(content: str) -> dict:
    """解析模型返回的文本，兼容 ```json ... ``` 包裹和多余文字。"""
    text = (content or "").strip()

    # 去掉 ```json ... ``` 代码块包裹
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()

    # 兜底：截取第一个 { 到最后一个 } 之间的内容
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]

    result = json.loads(text)

    if not isinstance(result, dict):
        raise ValueError("模型返回的不是 JSON 对象")

    return result


def _normalize(raw: dict) -> StudentInfo:
    return StudentInfo(
        name=_clean_str(raw.get("name")),
        age=_to_optional_int(raw.get("age")),
        major=_clean_str(raw.get("major")),
    )


# ==========================
# 接口
# ==========================

@router.post("/ai/extract-student", response_model=StudentInfo)
def extract_student(data: ExtractRequest):

    text = data.text.strip()

    # 输入为空（包括全是空格）
    if not text:
        raise HTTPException(
            status_code=400,
            detail="输入文本不能为空"
        )

    try:
        response = get_client().chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            # DeepSeek 的 JSON 模式，保证返回可解析的 JSON
            response_format={"type": "json_object"},
            temperature=0,
        )

    except HTTPException:
        # 例如未配置 API Key：原样抛出，提示中不含密钥
        raise

    except Exception:
        # 记录服务端日志，但不把内部细节返回给客户端
        logger.exception("调用大模型失败")

        raise HTTPException(
            status_code=502,
            detail="调用大模型失败，请检查 API 配置、网络或账户状态"
        )

    # 模型理论上会返回 choices，但空列表时必须兜住
    # 否则下面取 choices[0] 会抛未捕获的 IndexError，变成 500
    if not getattr(response, "choices", None):
        logger.warning("大模型返回了空的 choices")

        raise HTTPException(
            status_code=502,
            detail="大模型没有返回有效结果，请稍后重试"
        )

    content = response.choices[0].message.content

    try:
        raw = _parse_json(content)

    except (json.JSONDecodeError, ValueError):
        logger.warning("模型返回内容无法解析为 JSON: %r", content)

        raise HTTPException(
            status_code=502,
            detail="大模型返回的内容不是合法 JSON，请稍后重试"
        )

    return _normalize(raw)
