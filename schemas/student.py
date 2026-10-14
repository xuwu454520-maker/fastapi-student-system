from pydantic import BaseModel



class StudentCreate(BaseModel):

    name:str

    age:int

    major:str