import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


# ==========================================
# 1. 初始化 DeepSeek
# ==========================================

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError(
        "没有找到 DEEPSEEK_API_KEY，请检查项目根目录下的 .env 文件"
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)


# ==========================================
# 2. Memory：保存不同会话的对话历史
# ==========================================

MEMORY: dict[str, list[dict[str, str]]] = {}

MAX_HISTORY_MESSAGES = 12


def get_memory(session_id: str) -> list[dict[str, str]]:
    """读取指定会话的历史记录。"""
    if session_id not in MEMORY:
        MEMORY[session_id] = []

    return MEMORY[session_id]


def save_memory(
    session_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """保存本轮用户问题和 Agent 回答。"""
    history = get_memory(session_id)

    history.append({
        "role": "user",
        "content": user_message,
    })

    history.append({
        "role": "assistant",
        "content": assistant_message,
    })

    # 只保留最近 12 条消息
    MEMORY[session_id] = history[-MAX_HISTORY_MESSAGES:]


# ==========================================
# 3. State：当前任务的执行状态
# ==========================================

@dataclass
class WorkflowState:
    session_id: str
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: str = "pending"
    current_step: str = "start"
    tool_calls: int = 0
    retry_count: int = 0
    errors: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)


# ==========================================
# 4. 定义 Agent 可以使用的工具
# ==========================================

def add_numbers(a: float, b: float) -> dict[str, Any]:
    """计算两个数字的和。"""
    return {
        "success": True,
        "result": a + b,
    }


def search_knowledge(question: str) -> dict[str, Any]:
    """在本地知识文档中搜索相关内容。"""
    docs_dir = "rag_docs"

    if not os.path.isdir(docs_dir):
        return {
            "success": False,
            "message": "未找到 rag_docs 文件夹，请先创建知识文档目录。",
        }

    matches = []

    for filename in os.listdir(docs_dir):
        if not filename.lower().endswith(".txt"):
            continue

        path = os.path.join(docs_dir, filename)

        try:
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()

            # 简单关键词检索：仅作为 RAG 前的基础练习
            keywords = [
                word.strip()
                for word in question.replace("？", " ").replace("?", " ").split()
                if len(word.strip()) >= 2
            ]

            if any(word in content for word in keywords):
                matches.append({
                    "file": filename,
                    "content": content[:2000],
                })

        except (OSError, UnicodeError) as exc:
            matches.append({
                "file": filename,
                "error": str(exc),
            })

    if not matches:
        return {
            "success": True,
            "message": "没有找到明显匹配的文档内容。",
            "results": [],
        }

    return {
        "success": True,
        "results": matches[:3],
    }


def get_student_count() -> dict[str, Any]:
    """
    演示用的模拟工具。
    当前不连接 MySQL，返回的数字不是真实学生总数。
    """
    return {
        "success": True,
        "student_count": 3,
        "source": "mock_data",
    }


TOOL_FUNCTIONS = {
    "add_numbers": add_numbers,
    "search_knowledge": search_knowledge,
    "get_student_count": get_student_count,
}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "计算两个数字的和",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "检索本地 rag_docs 文件夹中的知识文档",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_student_count",
            "description": "查询演示用的学生数量，当前返回模拟数据",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


# ==========================================
# 5. Retry：工具失败后自动重试
# ==========================================

def execute_with_retry(
    tool_name: str,
    arguments: dict[str, Any],
    state: WorkflowState,
    max_retries: int = 2,
) -> dict[str, Any]:
    """
    最多执行 3 次：
    首次执行 1 次 + 失败重试 2 次。
    """
    function = TOOL_FUNCTIONS.get(tool_name)

    if function is None:
        return {
            "success": False,
            "error": f"未知工具：{tool_name}",
        }

    for attempt in range(max_retries + 1):
        try:
            result = function(**arguments)

            if not isinstance(result, dict):
                raise TypeError("工具必须返回字典类型的结果")

            if result.get("success") is False:
                raise RuntimeError(
                    result.get("message", "工具返回失败状态")
                )

            return result

        except Exception as exc:
            state.retry_count += 1
            state.errors.append(
                f"{tool_name} 第 {attempt + 1} 次执行失败：{exc}"
            )

            if attempt < max_retries:
                wait_seconds = 2 ** attempt
                print(
                    f"工具执行失败，"
                    f"{wait_seconds} 秒后重试……"
                )
                time.sleep(wait_seconds)
            else:
                return {
                    "success": False,
                    "error": str(exc),
                    "attempts": max_retries + 1,
                }

    return {
        "success": False,
        "error": "工具执行失败",
    }


# ==========================================
# 6. Workflow：按步骤执行 Agent 任务
# ==========================================

def run_agent(
    user_message: str,
    session_id: str,
) -> tuple[str, WorkflowState]:

    state = WorkflowState(session_id=session_id)
    state.status = "running"
    state.current_step = "load_memory"

    history = get_memory(session_id)

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个可靠的 Python 后端学习助手。"
                "需要计算时使用 add_numbers；"
                "需要查询知识时使用 search_knowledge；"
                "需要查询学生数量时使用 get_student_count。"
                "工具返回失败时，不要编造结果。"
                "学生数量工具目前返回模拟数据，必须明确说明。"
                "回答要清楚、简洁，并优先使用中文。"
            ),
        },
        *history,
        {
            "role": "user",
            "content": user_message,
        },
    ]

    try:
        state.current_step = "call_llm"

        # 限制 Agent 最多进行 5 轮模型调用
        for _ in range(5):
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
            )

            message = response.choices[0].message

            # 模型不需要调用工具，直接返回答案
            if not message.tool_calls:
                answer = message.content or "模型没有返回文本答案。"

                state.status = "completed"
                state.current_step = "finished"

                save_memory(
                    session_id,
                    user_message,
                    answer,
                )

                return answer, state

            # 先将模型的工具调用消息放进上下文
            messages.append(
                message.model_dump(exclude_none=True)
            )

            state.current_step = "execute_tools"

            for tool_call in message.tool_calls:
                state.tool_calls += 1

                tool_name = tool_call.function.name

                try:
                    arguments = json.loads(
                        tool_call.function.arguments or "{}"
                    )
                except json.JSONDecodeError as exc:
                    result = {
                        "success": False,
                        "error": f"工具参数不是有效 JSON：{exc}",
                    }
                else:
                    result = execute_with_retry(
                        tool_name,
                        arguments,
                        state,
                    )

                state.results.append(
                    json.dumps(result, ensure_ascii=False)
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        result,
                        ensure_ascii=False,
                    ),
                })

        raise RuntimeError(
            "超过最大模型调用轮数，工作流已停止。"
        )

    except Exception as exc:
        state.status = "failed"
        state.current_step = "error"
        state.errors.append(str(exc))

        answer = (
            "本次任务执行失败，未能可靠地完成回答。\n"
            f"错误信息：{exc}"
        )

        # 失败结果也写入会话历史，方便后续排查
        save_memory(
            session_id,
            user_message,
            answer,
        )

        return answer, state


# ==========================================
# 7. 命令行入口
# ==========================================

def main():
    session_id = str(uuid.uuid4())

    print("=" * 50)
    print("Day 26 Agent Workflow 已启动")
    print("命令：/new 新建会话；/exit 退出")
    print("=" * 50)

    while True:
        try:
            user_message = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAgent 已退出。")
            break

        if not user_message:
            continue

        if user_message.lower() == "/exit":
            print("Agent 已退出。")
            break

        if user_message.lower() == "/new":
            session_id = str(uuid.uuid4())
            print(f"已创建新会话：{session_id[:8]}")
            continue

        answer, state = run_agent(
            user_message,
            session_id,
        )

        print(f"\nAgent：{answer}")

        print("\n--- Workflow 状态 ---")
        print(f"任务 ID：{state.task_id}")
        print(f"会话 ID：{state.session_id[:8]}")
        print(f"任务状态：{state.status}")
        print(f"当前步骤：{state.current_step}")
        print(f"工具调用次数：{state.tool_calls}")
        print(f"累计重试次数：{state.retry_count}")

        if state.errors:
            print("错误记录：")
            for error in state.errors:
                print(f"- {error}")


if __name__ == "__main__":
    main()
