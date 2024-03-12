"""
DEPRECATED
"""

from fastapi import FastAPI

from sciencenow.api.arxiv import router
from sciencenow.db.db_setup import engine
from sciencenow.db.db_models import ArxivModel

ArxivModel.metadata.create_all(bind=engine)

# from sciencenow.db_models import ArxivPaper

app = FastAPI(
    title="ScienceNOW API",
    description="API to analyze Arxiv e-prints",
    version="0.0.1",
    contact={"name": "Bene Arnthof", "email": "benearnthof@hotmail.de"},
)

app.include_router(router)
