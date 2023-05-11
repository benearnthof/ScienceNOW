"""
Implementing Source Object for Reddit.com
"""

import praw
from datetime import datetime, timezone

from base import Source

class RedditSource(Source):
    def __init__(self, client_id, client_secret, user_agent, username, password):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            username=username,
            password=password,
        )
        self.last_fetch_time = datetime.now(timezone.utc)
    
    def get_posts(self, keywords=None, since=None, start=0, num=100):
        posts = []
        query = " ".join(keywords) if keywords else ""
        for submission in self.reddit.subreddit("all").search(
            query, sort="new", time_filter="day", limit=num, params={"count": start}
        ):
            post = {
                "id": submission.id,
                "title": submission.title,
                "author": submission.author.name,
                "url": submission.url,
                "created_utc": datetime.fromtimestamp(submission.created_utc, timezone.utc),
                "text": submission.selftext,
            }
            posts.append(post)
        return posts
    
    def fetch_new(self):
        posts = []
        for submission in self.reddit.subreddit("all").new(limit=None):
            if submission.created_utc <= self.last_fetch_time.timestamp():
                continue
            post = {
                "id": submission.id,
                "title": submission.title,
                "author": submission.author.name,
                "url": submission.url,
                "created_utc": datetime.fromtimestamp(submission.created_utc, timezone.utc),
                "text": submission.selftext,
            }
            posts.append(post)
        self.last_fetch_time = datetime.now(timezone.utc)
        return posts
    
    def fetch_all(self):
        posts = []
        for submission in self.reddit.subreddit("all").new(limit=None):
            post = {
                "id": submission.id,
                "title": submission.title,
                "author": submission.author.name,
                "url": submission.url,
                "created_utc": datetime.fromtimestamp(submission.created_utc, timezone.utc),
                "text": submission.selftext,
            }
            posts.append(post)
        return posts
