

from fastapi import FastAPI

from routers import student
from routers import auth


app = FastAPI(
    title="学生管理系统"
)



app.include_router(
    student.router
)


app.include_router(
    auth.router
)



@app.get("/")
def home():

    return {

        "message":"FastAPI项目启动成功"

    }