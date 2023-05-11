"""
Implementing Source Object for Twitter.com
"""

import tweepy
import time
from datetime import datetime, timezone
from pytz import timezone as tz

from base import Source

class TwitterSource(Source):
    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret):
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth)
        self.latest_id = None

    def get_posts(self, keywords=None, since=None, start=0, num=100):
        if keywords is None:
            keywords = ['python']  # Default keyword search
        q = ' OR '.join(keywords) + ' -filter:retweets'
        results = self.api.search(q=q, lang='en', result_type='recent', count=num)
        posts = []
        for r in results:
            post = {
                'id': r.id_str,
                'text': r.text,
                'created_at': r.created_at,
                'source': 'twitter.com',
                'author': r.author.name,
                'author_id': r.author.id_str,
                'url': f"https://twitter.com/{r.author.screen_name}/status/{r.id_str}"
            }
            posts.append(post)
        return posts

    def fetch_new(self):
        while True:
            if self.latest_id is None:
                # Get latest tweet if no tweet has been fetched before
                tweets = self.api.user_timeline(count=1)
            else:
                tweets = self.api.user_timeline(since_id=self.latest_id)
            if len(tweets) > 0:
                for t in reversed(tweets):
                    post = {
                        'id': t.id_str,
                        'text': t.text,
                        'created_at': t.created_at,
                        'source': 'twitter.com',
                        'author': t.author.name,
                        'author_id': t.author.id_str,
                        'url': f"https://twitter.com/{t.author.screen_name}/status/{t.id_str}"
                    }
                    self.latest_id = t.id_str
                    yield post
            time.sleep(30)

    def fetch_all(self):
        for status in tweepy.Cursor(self.api.user_timeline).items():
            post = {
                'id': status.id_str,
                'text': status.text,
                'created_at': status.created_at,
                'source': 'twitter.com',
                'author': status.author.name,
                'author_id': status.author.id_str,
                'url': f"https://twitter.com/{status.author.screen_name}/status/{status.id_str}"
            }
            yield post
