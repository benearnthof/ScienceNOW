"""
load environment variables and export them as constants for database, api tokens & paths
"""
import os
from dotenv import load_dotenv


PROJECT_ROOT = "envvars.env"

load_dotenv(PROJECT_ROOT)

DATABASE_HOSTNAME = os.environ.get("DB_HOSTNAME", "localhost")
DATABASE_NAME = os.environ.get("DB_DATABASE_NAME", "scienceNOW")
DATABASE_USER = os.environ.get("DB_USER", "user")
DATABASE_PASSWORD = os.environ.get("DB_PASSWORD", "pass")

# postgresql+psycopg2://user:password@hostname/database_name
DATABASE_URL = f"postgresql+psycopg2://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOSTNAME}/{DATABASE_NAME}"

# TWITTER_CONSUMER_KEY = os.environ.get('TWITTER_CONSUMER_KEY', "")
# TWITTER_CONSUMER_SECRET = os.environ.get('TWITTER_CONSUMER_SECRET', "")
# TWITTER_ACCESS_TOKEN = os.environ.get('TWITTER_ACCESS_TOKEN', "")
# TWITTER_ACCESS_SECRET = os.environ.get('TWITTER_ACCESS_SECRET', "")
