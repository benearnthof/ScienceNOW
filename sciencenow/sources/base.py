"""
Abstract base classes for all source objects
"""

from abc import ABCMeta, abstractmethod


class Source(metaclass=ABCMeta):
    @abstractmethod
    def get_posts(self, keywords=None, since=None, start=0, num=100):
        """
        Get a fixed number of recent posts.
        """

    @abstractmethod
    def fetch_new(self):
        """
        Fetch all new posts since this method was last run.
        """

    @abstractmethod
    def fetch_all(self):
        """
        Fetch all resources needed to instantiate the database.
        """
