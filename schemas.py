# app/schemas.py
# Описания форматов входящего и входящего JSON
from pydantic import BaseModel, Field

class RequestIn(BaseModel):
    query: str

class RequestOut(BaseModel):
    answer: str