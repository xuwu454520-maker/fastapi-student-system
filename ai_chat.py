import os

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field


# 读取 .env 文件里的配置
# 明确指定 .env 的路径
# 这样不管在哪个目录启动服务，都能读到配置
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

load_dotenv(
    os.path.join(BASE_DIR, ".env")
)


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

DEEPSEEK_BASE_URL = os.getenv(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com"
)

DEEPSEEK_MODEL = os.getenv(
    "DEEPSEEK_MODEL",
    "deepseek-chat"
)


# DeepSeek 兼容 OpenAI 接口格式
# 所以直接用 openai 这个库，只需要把 base_url 换成 DeepSeek 的地址
client = None


def get_client() -> OpenAI:

    global client

    if client is None:

        # 没有配置密钥时给出清楚的提示
        # 而不是让项目一启动就报错
        if not DEEPSEEK_API_KEY:

            raise HTTPException(
                status_code=500,
                detail="未配置 DEEPSEEK_API_KEY，请先在 .env 文件中填写"
            )

        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            timeout=30
        )

    return client


router = APIRouter(
    tags=["AI 对话"]
)


# 系统提示词：规定大模型扮演什么角色、用什么风格回答
SYSTEM_PROMPT = (
    "你是一个乐于助人的中文 AI 助手，"
    "回答要简洁、准确、通俗易懂。"
)


# ==========================
# 请求模型
# ==========================

class ChatRequest(BaseModel):

    message: str = Field(
        min_length=1,
        max_length=2000,
        description="用户发送的问题"
    )


# ==========================
# 响应模型
# ==========================

class ChatResponse(BaseModel):

    reply: str = Field(
        description="大模型生成的回答"
    )


# ==========================
# AI 对话接口
# ==========================

@router.post("/chat", response_model=ChatResponse)
def chat(data: ChatRequest):

    # 这里用同步的 def 而不是 async def
    # 因为调用大模型是阻塞操作
    # FastAPI 会自动把它放到线程池里执行，不会卡住其他请求

    response = get_client().chat.completions.create(

        model=DEEPSEEK_MODEL,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": data.message
            }
        ]
    )

    return ChatResponse(
        reply=response.choices[0].message.content
    )
