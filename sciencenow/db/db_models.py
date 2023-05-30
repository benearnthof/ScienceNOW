"""
SQLAlchemy Models for each website of interest
"""
import sys
from sqlalchemy import Column, Integer, String, Text, DateTime, Unicode, Boolean

from sqlalchemy_searchable import make_searchable
from sqlalchemy_utils.types import TSVectorType
from sqlalchemy_utils import URLType

from sciencenow.db.mixins import Timestamp
from sciencenow.db.db_setup import Base


def str_repr(string):
    if sys.version_info.major == 3:
        return string
    else:
        return string.encode("utf-8")


class ArxivModel(Timestamp, Base):
    """
    Arxiv Database Model for Postgres
    This is not equal to the pydantic model we use to create the API
    """

    __tablename__ = "arxiv"
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(Integer)
    popularity = Column(Integer)
    title = Column(Unicode(800, collation=""))
    arxiv_url = Column(URLType, primary_key=True)
    pdf_url = Column(URLType)
    published_time = Column(DateTime())
    authors = Column(Unicode(800, collation=""))
    abstract = Column(Text(collation=""))
    journal_link = Column(Text(collation=""), nullable=True)
    tag = Column(String(255))
    introduction = Column(Text(collation=""))
    conclusion = Column(Text(collation=""))
    analyzed = Column(Boolean, server_default="false", default=False)
    # For full text search
    search_vector = Column(
        TSVectorType(
            "title",
            "abstract",
            "authors",
            weights={"title": "A", "abstract": "B", "authors": "C"},
        )
    )

    def __repr__(self):
        template = '<Arxiv(id="{0}", url="{1}")>'
        return str_repr(template.format(self.id, self.arxiv_url))


class TwitterModel(Base):
    __tablename__ = "twitter_tweets"
    id = Column(String, primary_key=True)
    text = Column(String)
    author = Column(String)
    created_at = Column(Integer)


class RedditModel(Base):
    __tablename__ = "reddit_posts"
    id = Column(String, primary_key=True)
    title = Column(String)
    author = Column(String)
    subreddit = Column(String)
    created_utc = Column(Integer)
