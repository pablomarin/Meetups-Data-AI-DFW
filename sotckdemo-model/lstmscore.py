
# import the libraries
import keras
import tensorflow
import json
import shutil
import numpy as np


def init():
    # read in the model file
    from keras.models import model_from_json
    global loaded_model
    
    # load json and create model
    with open('modellstm.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights("modellstm.h5")

def run(score_input): 
    
    amount_of_features = len(score_input.columns)
    data = score_input.as_matrix() #converts to numpy
    seq_len = 10
    result = []
    for index in range(len(data) - seq_len):
        result.append(data[index: index + seq_len])

    result = np.array(result)

    seq_array = np.reshape(result, (result.shape[0], result.shape[1], amount_of_features))  
    
    try:
        prediction = loaded_model.predict_proba(seq_array)
        print(prediction)
        pred = prediction.tolist()
        return(pred)
    except Exception as e:
        return(str(e))
    
if __name__ == "__main__":
    init()
    run("{\"score_df\": [{\"Close\": 0.9403669834136963, \"High\": 0.7314814925193787, \"Open\": 0.7127071619033813, \"Volume\": 1.0, \"Low\": 0.5424528121948242}]}")