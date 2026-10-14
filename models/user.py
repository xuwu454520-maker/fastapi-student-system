from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String

from database import Base



class UserDB(Base):

    __tablename__ = "users"


    id = Column(
        Integer,
        primary_key=True,
        index=True
    )


    username = Column(
        String(50),
        unique=True,
        nullable=False
    )


    hashed_password = Column(
        String(255),
        nullable=False
    )