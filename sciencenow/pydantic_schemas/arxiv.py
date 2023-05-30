"""
Pydantic Schemas that match the db models
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ArxivBase(BaseModel):
    """
    ArxivPaper Model that inherits from pydantic.Basemodel that is needed to create the API.
    Only contains the base fields necessary to populate an "empty" paper entry.
    """

    title: str
    authors: Optional[str]
    abstract: Optional[str]
    arxiv_url: Optional[str]
    pdf_url: Optional[str]


class ArxivCreate(ArxivBase):
    ...


class Arxiv(ArxivBase):
    id: int
    version: int
    popularity: int
    published_time: datetime
    journal_link: Optional[str]
    tag: str
    introduction: Optional[str]
    conclusion: Optional[str]
    analyzed: bool
    search_vector: Optional[str]

    class Config:
        orm_mode = True
