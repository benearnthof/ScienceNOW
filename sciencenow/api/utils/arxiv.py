"""
Utilities to allow the api to interact with the database
"""

from sqlalchemy.orm import Session
from sciencenow.db.db_models import ArxivModel  # user
from sciencenow.pydantic_schemas.arxiv import ArxivCreate  # usercreate


def get_paper(db: Session, paper_id: int):
    return db.query(ArxivModel).filter(ArxivModel.id == paper_id).first()


def get_paper_by_title(db: Session, title: str):
    return db.query(ArxivModel).filter(ArxivModel.title == title).first()


def get_papers(db: Session, skip: int = 0, limit: int = 100):
    return db.query(ArxivModel).offset(skip).limit(limit).all()


def create_paper(db: Session, paper: ArxivCreate):
    db_paper = ArxivModel(
        title=paper.title,
        authors=paper.authors,
        abstract=paper.abstract,
        arxiv_url=paper.arxiv_url,
        pdf_url=paper.pdf_url,
    )
    db.add(db_paper)
    db.commit()
    db.refresh(db_paper)
    return db_paper
