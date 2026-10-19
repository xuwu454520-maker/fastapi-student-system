import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 1. 读取 .env 配置
load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("没有找到 DEEPSEEK_API_KEY，请检查 .env 文件")

# 2. 创建 DeepSeek 客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-chat"


# 3. 定义真正由 Python 执行的工具函数
def get_student_count():
    """查询学生总数。这里使用模拟数据。"""
    students = [
        {"name": "张三", "major": "人工智能"},
        {"name": "李四", "major": "计算机科学"},
        {"name": "王五", "major": "软件工程"},
    ]

    return {
        "total": len(students),
        "source": "mock_data",
    }


def add_numbers(a, b):
    """计算两个数字的和。"""
    return {
        "a": a,
        "b": b,
        "result": a + b,
    }


# 4. 向模型声明有哪些工具、工具参数是什么
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_student_count",
            "description": "查询当前模拟学生数据中的学生总数",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "计算两个数字的和",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "第一个数字",
                    },
                    "b": {
                        "type": "number",
                        "description": "第二个数字",
                    },
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    },
]


# 5. 建立工具名称与 Python 函数的对应关系
available_functions = {
    "get_student_count": get_student_count,
    "add_numbers": add_numbers,
}


def ask_ai(user_message):
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个助手。需要计算或查询学生数量时，"
                "请调用提供的工具。根据工具返回的真实结果回答，"
                "不要编造工具执行结果。"
            ),
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    # 6. 先让模型判断是否需要调用工具
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    assistant_message = response.choices[0].message
    messages.append(assistant_message)

    # 7. 如果模型要求调用工具，就由 Python 执行
    if assistant_message.tool_calls:
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(
                tool_call.function.arguments or "{}"
            )

            print(f"\n调用工具：{function_name}")
            print(f"调用参数：{arguments}")

            if function_name not in available_functions:
                result = {"error": "不允许调用该工具"}
            else:
                try:
                    function = available_functions[function_name]
                    result = function(**arguments)
                except (TypeError, ValueError) as exc:
                    result = {"error": f"参数错误：{exc}"}
                except Exception:
                    result = {"error": "工具执行失败"}

            print(f"工具结果：{result}")

            # 8. 将工具执行结果交还给模型
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(
                    result,
                    ensure_ascii=False,
                ),
            })

        # 9. 模型根据工具结果生成最终回答
        final_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )

        return final_response.choices[0].message.content

    # 模型认为不需要工具时，直接返回回答
    return assistant_message.content


if __name__ == "__main__":
    while True:
        question = input("\n请输入问题（输入 q 退出）：").strip()

        if question.lower() == "q":
            break

        if not question:
            print("问题不能为空")
            continue

        try:
            answer = ask_ai(question)
            print(f"\nAI 回答：{answer}")
        except Exception as exc:
            print(f"\n请求失败：{exc}")
