from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String

from database import Base



class StudentDB(Base):

    __tablename__ = "students"


    id = Column(
        Integer,
        primary_key=True,
        index=True
    )


    name = Column(
        String(50),
        nullable=False
    )


    age = Column(
        Integer,
        nullable=False
    )


    major = Column(
        String(100)
    )