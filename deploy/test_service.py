
import urllib
import json 
import requests
import pandas as pd

# The URL will need to be editted after service create.
url = 'http://40.76.29.12:5001/score'

## Sequence length will need to match the training sequence length from the model training
sequence_length = 10

# We'll read in this data to test the service
body = pd.read_pickle('test_dataframe.pkl')
headers = {'Content-Type':'application/json'}

try:
    if body.shape[0] < sequence_length : 
        print("Skipping scoring as we need {} records to score and only have {} records.".format(sequence_length, body.shape[0]))
    else:
        #print('{}'.format(body.shape))
        body = json.dumps({"data": body.to_json(orient='records')})
        print (body + '\n')
        req = urllib.request.Request(url, str.encode(body), headers) 
        
        with urllib.request.urlopen(req) as response:
            the_page = response.read()
            print('{}'.format(the_page))
        
except urllib.error.HTTPError as error:
    print("The request failed with status code {}: \n{}".format(error, error.read))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.reason)      