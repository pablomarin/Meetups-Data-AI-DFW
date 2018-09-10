
import urllib
import json 
import requests
import pandas as pd
import sys

# The URL will need to be editted after service create.
url =  sys.argv[1] # gets the url from the console argument 

## Sequence length will need to match the training sequence length from the model training
sequence_length = 10

# We'll read in this data to test the service
body = '{"data": "[{\\"Close\\":276.56,\\"sma_diff\\":0.1256666667,\\"ema_diff\\":-0.9208042383,\\"rsi\\":42.9571116039},{\\"Close\\":275.5,\\"sma_diff\\":1.2423333333,\\"ema_diff\\":-0.5204372189,\\"rsi\\":33.4023302607},{\\"Close\\":275.97,\\"sma_diff\\":1.486,\\"ema_diff\\":-0.3212150757,\\"rsi\\":40.7113516968},{\\"Close\\":274.24,\\"sma_diff\\":1.9463333333,\\"ema_diff\\":0.0610113416,\\"rsi\\":27.0514200019},{\\"Close\\":274.74,\\"sma_diff\\":1.8546666667,\\"ema_diff\\":0.2211086852,\\"rsi\\":34.9380909623},{\\"Close\\":271.0,\\"sma_diff\\":2.7553333333,\\"ema_diff\\":0.844388337,\\"rsi\\":17.3747515618},{\\"Close\\":271.6,\\"sma_diff\\":2.9033333333,\\"ema_diff\\":1.105809008,\\"rsi\\":24.9413127492},{\\"Close\\":269.35,\\"sma_diff\\":3.832,\\"ema_diff\\":1.5421559899,\\"rsi\\":17.4504277746},{\\"Close\\":270.89,\\"sma_diff\\":3.0846666667,\\"ema_diff\\":1.5263620609,\\"rsi\\":34.3259071505},{\\"Close\\":271.28,\\"sma_diff\\":2.6063333333,\\"ema_diff\\":1.3986775304,\\"rsi\\":38.3175942846}]"}'
headers = {'Content-Type':'application/json'}

try:
    if body.shape[0] < sequence_length : 
        print("Skipping scoring as we need {} records to score and only have {} records.".format(sequence_length, body.shape[0]))
    else:
        #print('{}'.format(body.shape))
        body = json.dumps({"data": body.to_json(orient='records')})
        req = urllib.request.Request(url, str.encode(body), headers) 
        
        with urllib.request.urlopen(req) as response:
            the_page = response.read()
            print('{}'.format(the_page))
        
except urllib.error.HTTPError as error:
    print("The request failed with status code {}: \n{}".format(error, error.read))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.reason)      