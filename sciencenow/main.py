"""
FastAPI app that gets entries from Arxiv & saves them to a postgres db
"""
from typing import List, Optional
from fastapi import FastAPI, Path
from pydantic import BaseModel

from api.arxiv import router

# from sciencenow.db_models import ArxivPaper

app = FastAPI(
    title="ScienceNOW API",
    description="API to analyze Arxiv e-prints",
    version="0.0.1",
    contact={"name": "Bene Arnthof", "email": "benearnthof@hotmail.de"},
)

app.include_router(router)
