"""
Topic Modeling with BERTopic
"""

from ..sciencenow.db import session_scope
from ..db_models import ArxivModel

from bertopic import BERTopic

# TODO: implement fetch_new method that returns ArxivModel 


titles = []
abstracts = []

session = session_scope()

for row in session.query(ArxivModel).all():
    titles.append(row.title)
    abstracts.append(row.abstract)


# Create a BERTopic model
model = BERTopic()

# Fit the model on the data
docs = titles + abstracts
model.fit(docs)

# Get the top 10 topics and their most representative documents
topics = model.get_topic_freq()
for topic_id, freq in topics[:10]:
    representative_docs = model.get_topic(topic_id)[:5]
    print(f"Topic {topic_id} ({freq} documents):")
    for doc in representative_docs:
        print(f"- {docs[doc]}")
