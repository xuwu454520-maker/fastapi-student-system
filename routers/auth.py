from datetime import datetime, timedelta, timezone

import jwt
from pwdlib import PasswordHash
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import Session

from database import Base, get_db
from models.user import UserDB


# 请替换成第三步生成的随机密钥
SECRET_KEY = "5bc98f96885ae9797bd19024ef38e13efabe936fa4385f1979f3160bd7dd9365"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

router = APIRouter(prefix="/auth", tags=["用户认证"])

password_hash = PasswordHash.recommended()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")




# 注册请求模型
class UserRegister(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=128)


# 对密码进行哈希，不保存明文密码
def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return password_hash.verify(password, hashed_password)


# 创建 JWT Token
def create_access_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": username,
        "exp": expire,
    }

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


# 从 Token 中识别当前用户
def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token 无效或已过期，请重新登录",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
        )
        username = payload.get("sub")

        if not username:
            raise credentials_error

    except jwt.InvalidTokenError:
        raise credentials_error

    user = (
        db.query(UserDB)
        .filter(UserDB.username == username)
        .first()
    )

    if user is None:
        raise credentials_error

    return user


# 用户注册
@router.post("/register")
def register(
    data: UserRegister,
    db: Session = Depends(get_db),
):
    existing_user = (
        db.query(UserDB)
        .filter(UserDB.username == data.username)
        .first()
    )

    if existing_user:
        raise HTTPException(
            status_code=409,
            detail="用户名已存在",
        )

    user = UserDB(
        username=data.username,
        hashed_password=hash_password(data.password),
    )

    db.add(user)

    try:
        db.commit()
        db.refresh(user)
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="注册失败，请检查数据库配置",
        )

    return {
        "message": "注册成功",
        "id": user.id,
        "username": user.username,
    }


# 用户登录：使用表单格式提交用户名和密码
@router.post("/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = (
        db.query(UserDB)
        .filter(UserDB.username == form_data.username)
        .first()
    )

    if (
        user is None
        or not verify_password(
            form_data.password,
            user.hashed_password,
        )
    ):
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(user.username)

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }


# 受保护的用户信息接口
@router.get("/me")
def read_current_user(
    current_user: UserDB = Depends(get_current_user),
):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "message": "Token 验证成功",
    }