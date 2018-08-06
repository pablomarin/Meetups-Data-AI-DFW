
import pickle
import json
import numpy as np
import pandas as pd
from azureml.core.model import Model
from keras.models import load_model

TICKER = "SPY"
MODEL = TICKER +'-modellstm.h5'
MIN_MAX_DICT = TICKER +'-min_max.pkl'


def init():
    global model
    global min_max_dict_list
    
    # load model
    model_path = Model.get_model_path(model_name = LSTM_MODEL)
    model = load_model(model_path)

    # Load Min Max list values
    model_path = Model.get_model_path(model_name = MIN_MAX_DICT)
    with open(model_path, 'rb') as handle:
        min_max_dict_list = pickle.load(handle)
        print("Min_max List loaded")

def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = pd.read_json(data, orient='records')
        data_n = data.copy()
        
        # Normalize data
        min_dict = min_max_dict_list[0]
        max_dict = min_max_dict_list[1]
        for feature_name in data.columns:
            data_n[feature_name] = (data[feature_name] - min_dict[feature_name]) / (max_dict[feature_name] - min_dict[feature_name])
        
        # Create sequences
        data = data_n.values 
        seq_len = 10
        result = []
        for index in range(len(data) - seq_len + 1):
            result.append(data[index: index + seq_len])

        result = np.array(result)
        print(result.shape)
        
        pred = model.predict(result)
        print(pred)
        
        # De-normalize the target
        pred = pred * (max_dict["Close"] - min_dict["Close"]) + min_dict["Close"]
        
        # Send results
        pred = pred.tolist()
        return json.dumps({"result": pred})

    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})