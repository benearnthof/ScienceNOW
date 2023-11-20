"""
Deprecated.
"""

import time
from urllib.request import urlopen
from urllib.error import HTTPError
import datetime
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path


pd.set_option("mode.chained_assignment", "warn")

OAI = "{http://www.openarchives.org/OAI/2.0/}"
ARXIV = "{http://arxiv.org/OAI/arXiv/}"


def scrape(arxiv="cs"):
    i = 0
    df = pd.DataFrame(
        columns=("title", "abstract", "categories", "created", "id", "doi")
    )
    base_url = "http://export.arxiv.org/oai2?verb=ListRecords&"
    url = (
        base_url
        + "from=2020-01-01&until=2022-12-31&"
        + "metadataPrefix=arXiv&set=%s" % arxiv
    )
    while i < 10:
        print("fetching", url)
        try:
            response = urlopen(url)
        except HTTPError as e:
            if e.code == 503:
                to = int(e.hdrs.get("retry-after", 30))
                print("Got 503. Retrying after {0:d} seconds.".format(to))
                time.sleep(to)
                continue
            else:
                raise
        xml = response.read()
        root = ET.fromstring(xml)
        for record in root.find(OAI + "ListRecords").findall(OAI + "record"):
            arxiv_id = record.find(OAI + "header").find(OAI + "identifier")
            meta = record.find(OAI + "metadata")
            info = meta.find(ARXIV + "arXiv")
            created = info.find(ARXIV + "created").text
            created = datetime.datetime.strptime(created, "%Y-%m-%d")
            categories = info.find(ARXIV + "categories").text
            # if there is more than one DOI use the first one
            # often the second one (if it exists at all) refers
            # to an eratum or similar
            doi = info.find(ARXIV + "doi")
            if doi is not None:
                doi = doi.text.split()[0]
            contents = {
                "title": info.find(ARXIV + "title").text,
                "id": info.find(ARXIV + "id").text,  # arxiv_id.text[4:],
                "abstract": info.find(ARXIV + "abstract").text.strip(),
                "created": created,
                "categories": categories.split(),
                "doi": doi,
            }
            cont_df = pd.DataFrame(data=contents)
            df = pd.concat([df, cont_df], ignore_index=True)
        # The list of articles returned by the API comes in chunks of
        # 1000 articles. The presence of a resumptionToken tells us that
        # there is more to be fetched.
        token = root.find(OAI + "ListRecords").find(OAI + "resumptionToken")
        if token is None or token.text is None:
            break
        else:
            url = base_url + "resumptionToken=%s" % (token.text)
        i += 1
    return df


df = scrape()

store = pd.HDFStore(Path("c:/arxiv/arxiv_stream.h5"))
store["df"] = df
# df = store["df"]
store.close()
