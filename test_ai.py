# 单独测试能否调通 DeepSeek，不经过 FastAPI
from ai_chat import (
    DEEPSEEK_MODEL,
    SYSTEM_PROMPT,
    get_client,
)

question = "请用简单的语言解释什么是 FastAPI"

response = get_client().chat.completions.create(
    model=DEEPSEEK_MODEL,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ],
)

print("用户问题：", question)
print("AI回答：", response.choices[0].message.content)
