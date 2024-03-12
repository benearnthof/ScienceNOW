"""
DEPRECATED
"""

# need a login, will need to continuously scroll to load more data
import pandas as pd
from tqdm import tqdm
import snscrape.modules.twitter as sntwitter


# how do we best run this for a large number of papers?
# restricting ourselves to certain users/hashtags may be helpful
# could we run this in parallel?


# lets wrap it up
class Scraper:
    """
    Wrapper Class for Search, User, Profile, Hashtag, and Trends
    """

    def __init__(self, keyword, type, **kwargs):
        super().__init__(**kwargs)
        self.keyword = keyword
        self.type = type
        self.mode = sntwitter.TwitterSearchScraperMode.LIVE
        self.scraper = self._initscraper()

    def _initscraper(self):
        """
        Initializes the scraper
        """
        match self.type:
            case "search":
                return sntwitter.TwitterSearchScraper(
                    query=self.keyword, mode=self.mode
                )
            case "user":
                return sntwitter.TwitterUserScraper(user=self.keyword, mode=self.mode)
            case "profile":
                return sntwitter.TwitterProfileScraper(
                    user=self.keyword, mode=self.mode
                )
            case "hashtag":
                return sntwitter.TwitterHashtagScraper(
                    hashtag=self.keyword, mode=self.mode
                )

    def scrape(self, ntweets=999):
        results = []
        for i, tweet in tqdm(enumerate(self.scraper.get_items()), total=ntweets):
            data = [
                tweet.date,
                tweet.id,
                tweet.rawContent,
                tweet.user.username,
                tweet.likeCount,
                tweet.retweetCount,
            ]
            results.append(data)
            if i == ntweets:
                break
        out = pd.DataFrame(
            results,
            columns=["date", "id", "content", "username", "likecount", "retweetcount"],
        )
        return out
