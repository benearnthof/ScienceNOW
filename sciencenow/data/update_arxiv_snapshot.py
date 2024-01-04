import gzip
import json
import time
from pathlib import Path
from os import getcwd
from omegaconf import OmegaConf
from arxiv_public_data.oai_metadata import (
    all_of_arxiv, 
    get_list_record_chunk, 
    check_xml_errors, 
    parse_xml_listrecords,
    URL_ARXIV_OAI,
)
from warnings import warn
import requests
import xml.etree.ElementTree as ET

from arxiv_public_data.config import LOGGER, DIR_BASE

log = LOGGER.getChild('metadata')


cfg = "./sciencenow/config/secrets.yaml"
config = OmegaConf.load(cfg)

outfile = config.ARXIV_SNAPSHOT + ".gz"

tokenfile = '{}-resumptionToken.txt'.format(outfile)

if not Path(outfile).exists():
    print(f"Warning: {outfile} does not exist.")

if Path(tokenfile).exists():
    old_token = open(tokenfile, 'r').read()
else:
    warn("No resumption token found.")

def update_arxiv_snapshot(
    old_token,
    harvest_url=URL_ARXIV_OAI,
    metadataPrefix='arXivRaw',
    outfile=outfile):
    """
    Queries the OIA API for the metadata of 1 chunk of 1000 articles to obtain a new resumption token,
    Then uses the index of the old resumption token to continue the download of new chunks without having to 
    redownload the entire dataset from scratch.

    Parameters
    ----------
        old_token : str
            expired resumptionToken
    """
    parameters = {'verb': 'ListRecords'}
    parameters['metadataPrefix'] = metadataPrefix
    response = requests.get(harvest_url, params=parameters)
    xml_root = ET.fromstring(response.text)
    _, new_token = parse_xml_listrecords(xml_root)
    index = old_token.split("|")[1]
    new_token = new_token.split("|")[0] + "|" + index
    resumptionToken = new_token
    chunk_index = 0
    total_records = 0
    while True:
        log.info('Index {:4d} | Records {:7d} | resumptionToken "{}"'.format(
            chunk_index, total_records, resumptionToken)
        )
        xml_root = ET.fromstring(get_list_record_chunk(resumptionToken))
        check_xml_errors(xml_root)
        records, resumptionToken = parse_xml_listrecords(xml_root)
        chunk_index = chunk_index + 1
        total_records = total_records + len(records)
        with gzip.open(outfile, 'at', encoding='utf-8') as fout:
            for rec in records:
                fout.write(json.dumps(rec) + '\n')
        if resumptionToken:
            with open(tokenfile, 'w') as fout:
                fout.write(resumptionToken)
        else:
            log.info('No resumption token, query finished')
            return
        time.sleep(12)  # OAI server usually requires a 10s wait


# old_token = "6854384|2363001"

# update_arxiv_snapshot(old_token)

if __name__=='__main__':
    update_arxiv_snapshot(old_token)
