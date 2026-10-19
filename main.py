

from contextlib import asynccontextmanager

from fastapi import FastAPI

from routers import student
from routers import auth
from routers import agent
from routers.agent import agent_service

import ai_chat
import ai_extract


@asynccontextmanager
async def lifespan(app: FastAPI):
    await agent_service.start()
    try:
        yield
    finally:
        await agent_service.close()


app = FastAPI(
    title="学生管理系统",
    lifespan=lifespan,
)



app.include_router(
    student.router
)


app.include_router(
    auth.router
)


app.include_router(
    ai_chat.router
)


app.include_router(
    ai_extract.router
)


app.include_router(
    agent.router
)



@app.get("/")
def home():

    return {

        "message":"FastAPI项目启动成功"

    }