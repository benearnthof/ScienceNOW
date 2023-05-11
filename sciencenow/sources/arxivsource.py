"""
Implementing Source Object for Arxiv.com
"""

import feedparser
import time
import datetime
from urllib.parse import urlencode

class ArxivSource(Source):
    """
    Uses feedparser library to retrieve posts from the arxiv.com RSS feed. 
    """
    def __init__(self, base_url='http://export.arxiv.org/api/query?'):
      """
      Initializes the base_url attribute with the arxiv.com API endpoint.
      """
        self.base_url = base_url
    
    def get_posts(self, keywords=None, since=None, start=0, num=100):
        """
        Retrieve a fixed number of recent posts from arxiv.com.
        Constructs the query URL based on the provided criteria, retrieves the feed using feedparser.parse,
        and extracts relevant post information from the feed entries. 
        """
        # Construct query parameters
        query_params = {
            'search_query': 'all:{}'.format(keywords) if keywords else 'all',
            'start': start,
            'max_results': num,
            'sortBy': 'lastUpdatedDate',
            'sortOrder': 'descending',
            'timeframe': 'last_7_days' if since is None else 'last_30_days',
        }
        if since:
            query_params['since'] = since.strftime('%Y-%m-%d')
        
        # Construct query URL and retrieve feed
        url = self.base_url + urlencode(query_params)
        feed = feedparser.parse(url)
        
        # Extract relevant post information
        posts = []
        for entry in feed.entries:
            title = entry.title
            link = entry.link
            summary = entry.summary
            published = datetime.datetime.strptime(entry.published, '%Y-%m-%dT%H:%M:%SZ')
            updated = datetime.datetime.strptime(entry.updated, '%Y-%m-%dT%H:%M:%SZ')
            posts.append({
                'title': title,
                'link': link,
                'summary': summary,
                'published': published,
                'updated': updated,
            })
        
        return posts
    
    def fetch_new(self):
        """
        Retrieve all new posts since the last time this method was run.
        Gets the last fetch time from the database, sets the since parameter in the get_posts method to the last fetch time plus one second, 
        obtains new posts using get_posts, updates the database with the new posts, and updates the last fetch time in the database to the current time. 
        """
        # Get last fetch time from database
        last_fetch_time = # Get last fetch time from database
        
        # Set since parameter to last fetch time
        since = last_fetch_time + datetime.timedelta(seconds=1)
        
        # Retrieve new posts and update database
        new_posts = self.get_posts(since=since)
        # Update database with new posts
        
        # Update last fetch time in database
        current_time = datetime.datetime.now()
        # Update last fetch time in database to current_time
        
        return new_posts
    
    def fetch_all(self):
        """
        Retrieve all posts from arxiv.com.
        Calls `get_posts` without any criteria, updates the database with all posts, and returns all posts.
        """
        # Retrieve all posts and update database
        all_posts = self.get_posts()
        # TODO: Update database with all posts
        
        return all_posts
