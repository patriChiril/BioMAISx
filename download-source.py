"""Code for downloading articles from Factiva that compose source of BioMAISx.

Adapted from
https://developer.dowjones.com/site/docs/factiva_apis/factiva_analytics_apis/factiva_snapshots_api/index.gsp
The following code uses Snapshots to execute a query which prompts the user for
a search term and an article limit, and then submits that query.
"""

import json
import logging
import os
from time import sleep

import requests

logging.basicConfig(
    filename="example.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)

USER_KEY = os.environ["FACTIVA_USER_KEY"]
CID = os.environ["FACTIVA_CID"]

# The URL of the Extractions Endpoint
account_info_url = f"https://api.dowjones.com/alpha/accounts/{USER_KEY}"
search_url = "https://api.dowjones.com/alpha/extractions/documents"
explain_url = f"{search_url}/_explain"
search_query = (
    "(REGEXP_CONTAINS(CONCAT(title, ' ', IFNULL(snippet, ''), ' ', IFNULL(body, '')"
    ", ' ', IFNULL(section, '')), r'(?i)(\\b)(gmos|transgénèse|ogm|"
    "genetically\\W+modified\\W+organism\\w{0,2}|gm\\W+crop\\w{0,2}|transgénique\\"
    "w{0,1}|(\\b(patrimoine\\W+génétique|genome)\\W+(?:\\w+\\W+){0,3}?"
    "(modifié\\W+artificiellement)\\b)|(\\b(culture\\w{0,1})\\W+(?:\\w+\\W+){0,2}?"
    "(ogm|organisme\\w{0,1}\\W+génétiquement\\W+modifie\\w{0,1})\\b)|gmo|organisme"
    "\\w{0,1}\\W+génétiquement\\W+modifié\\w{0,1}|genetically\\W+modified\\W+crop\\w"
    "{0,2})(\\b)') OR REGEXP_CONTAINS(CONCAT(title, ' ', IFNULL(snippet, ''), ' ',"
    " IFNULL(body, ''), ' ', IFNULL(section, '')), r'(?i)(\\b)(agriculture|culture"
    "\\w{0,1}|agriculture|crop\\w{0,1})(\\b)') OR REGEXP_CONTAINS(industry_codes, r'"
    "(?i)(^|,)(igmfd)($|,)')) AND (REGEXP_CONTAINS(restrictor_codes, r'(?i)(^|,)"
    "(africa)($|,)') OR REGEXP_CONTAINS(region_of_origin, r'(?i)(africa)')) AND LOWER"
    "(language_code) IN ('fr', 'en')"
)
headers = {"content-type": "application/json", "user-key": USER_KEY}
request_body = {"query": {"where": search_query}}

# Create an explain with the given query
logging.info("Creating an explain: " + json.dumps(request_body))
response = requests.post(explain_url, data=json.dumps(request_body), headers=headers)

# Check the explain to verify the query was valid and see how many docs
# would be returned
if response.status_code != 201:
    raise RuntimeError("ERROR: An error occurred creating an explain: " + response.text)

explain = response.json()
logging.info("Explain Created. Job ID: " + explain["data"]["id"])
state = explain["data"]["attributes"]["current_state"]
logging.debug(response.json())

# wait for explain job to complete
while state != "JOB_STATE_DONE":
    self_link = explain["links"]["self"]
    response = requests.get(self_link, headers=headers)
    logging.debug(response.json())
    explain = response.json()
    state = explain["data"]["attributes"]["current_state"]
    sleep(15)

logging.info("Explain Completed Successfully.")
doc_count = explain["data"]["attributes"]["counts"]
logging.info("Number of documents returned: " + str(doc_count))

if int(doc_count) < 1900000 or int(doc_count) > 2000000:
    raise RuntimeError("Estimated number of documents should be ~1950000")

# Create a Snapshot with the given query
logging.info("Creating the Snapshot: " + json.dumps(request_body))
response = requests.post(
    search_url, data=json.dumps(request_body), headers=headers
)
logging.info(response.text)
logging.debug(response.json())

# Verify the response from creating an extraction is OK
if response.status_code != 201:
    raise RuntimeError(
        "ERROR: An error occurred creating an extraction: " + response.text
    )
extraction = response.json()
logging.info(extraction)
logging.info("Extraction Created. Job ID: " + extraction["data"]["id"])
self_link = extraction["links"]["self"]
sleep(30)
logging.info("Checking state of the job.")

while True:
    # We now call the second endpoint, which will tell us if the
    # extraction is ready.
    status_response = requests.get(self_link, headers=headers)
    logging.debug(status_response.json())

    # Verify the response from the self_link is OK
    if status_response.status_code != 200:
        raise RuntimeError(
            "ERROR: an error occurred getting the details "
            "for the extraction: " + status_response.text
        )
    # There is an edge case where the job does not have a
    # current_state yet. If current_state does not yet exist
    # in the response, we will sleep for 10 seconds
    status = status_response.json()

    if "current_state" in status["data"]["attributes"]:
        currentState = status["data"]["attributes"]["current_state"]
        logging.info("Current state is: " + currentState)

        # Job is still running, Sleep for 10 seconds
        if currentState == "JOB_STATE_RUNNING":
            logging.info("Sleep for 30 seconds - Job state running")
            sleep(30)

        elif currentState == "JOB_VALIDATING":
            logging.info("Sleep for 30 seconds - Job validating")
            sleep(30)

        elif currentState == "JOB_QUEUED":
            logging.info("Sleeping for 30 seconds... Job queued")
            sleep(30)

        elif currentState == "JOB_CREATED":
            logging.info("Sleeping for 30 seconds... Job created")
            sleep(30)

        else:
            # If currentState is JOB_STATE_DONE then everything
            # completed successfully
            if currentState == "JOB_STATE_DONE":
                logging.info("Job completed successfully")
                logging.info(
                    "Downloading Snapshot files to current directory"
                )
                for file in status["data"]["attributes"]["files"]:
                    filepath = file["uri"]
                    parts = filepath.split("/")
                    filename = parts[len(parts) - 1]
                    r = requests.get(
                        file["uri"], stream=True, headers=headers
                    )
                    dir_path = os.path.dirname(
                        os.path.realpath(__file__)
                    )
                    filename = os.path.join(dir_path, filename)
                    with open(filename, "wb") as fd:
                        for chunk in r.iter_content(chunk_size=128):
                            fd.write(chunk)

            # Job has another state that means it was not successful.
            else:
                logging.info(
                    "An error occurred with the job. Final state is: "
                    + currentState
                )

            break
    else:
        logging.info("Sleeping for 30 seconds...")
        sleep(30)
