from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base


DATABASE_URL = (
    "mysql+pymysql://root:454520@localhost:3306/school?charset=utf8mb4"
)


engine = create_engine(
    DATABASE_URL,
    echo=True,
    pool_pre_ping=True,
)


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


Base = declarative_base()


# 数据库依赖
def get_db():

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()