"""
SQLAlchemy Models for each website of interest
"""

from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class RedditPost(Base):
    __tablename__ = "reddit_posts"
    id = Column(String, primary_key=True)
    title = Column(String)
    author = Column(String)
    subreddit = Column(String)
    created_utc = Column(Integer)

class ArxivPaper(Base):
    __tablename__ = "arxiv_papers"
    id = Column(String, primary_key=True)
    title = Column(String)
    authors = Column(String)
    abstract = Column(String)
    categories = Column(String)
    created_utc = Column(Integer)

class TwitterTweet(Base):
    __tablename__ = "twitter_tweets"
    id = Column(String, primary_key=True)
    text = Column(String)
    author = Column(String)
    created_at = Column(Integer)

# TODO: Create the database engine and create the tables
