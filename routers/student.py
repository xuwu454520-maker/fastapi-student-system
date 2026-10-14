import json

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import get_db

from models.student import StudentDB

from schemas.student import StudentCreate

from redis_client import redis_client


router = APIRouter(
    prefix="/students",
    tags=["学生管理"]
)


# ==========================
# 新增学生
# ==========================

@router.post("/")
def create_student(
    student: StudentCreate,
    db: Session = Depends(get_db)
):

    new_student = StudentDB(
        name=student.name,
        age=student.age,
        major=student.major
    )


    db.add(new_student)

    db.commit()

    db.refresh(new_student)


    # 新增学生后删除缓存
    redis_client.delete(
        "students"
    )


    return {
        "message": "学生添加成功",
        "data": {
            "id": new_student.id,
            "name": new_student.name,
            "age": new_student.age,
            "major": new_student.major
        }
    }



# ==========================
# 查询学生（Redis缓存）
# ==========================

@router.get("/")
def get_students(
    db: Session = Depends(get_db)
):

    # 1. 查询 Redis 缓存

    cache_data = redis_client.get(
        "students"
    )


    if cache_data:

        return {
            "source": "redis",
            "data": json.loads(cache_data)
        }



    # 2. Redis没有数据
    # 查询 MySQL

    students = (
        db.query(StudentDB)
        .all()
    )


    result = []


    for student in students:

        result.append(
            {
                "id": student.id,
                "name": student.name,
                "age": student.age,
                "major": student.major
            }
        )



    # 3. 保存到 Redis

    redis_client.set(
        "students",
        json.dumps(result),
        ex=60
    )


    return {
        "source": "mysql",
        "data": result
    }



# ==========================
# 查询单个学生
# ==========================

@router.get("/{student_id}")
def get_student(
    student_id: int,
    db: Session = Depends(get_db)
):

    student = (
        db.query(StudentDB)
        .filter(
            StudentDB.id == student_id
        )
        .first()
    )


    return student