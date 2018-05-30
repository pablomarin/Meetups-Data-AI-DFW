
# import the libraries
import keras
import tensorflow
import json
import shutil
import numpy as np
import pandas as pd


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
    for index in range(len(data) - seq_len + 1):
        result.append(data[index: index + seq_len + 1])
        
    result = np.array(result)
    
    seq_array = np.reshape(result, (result.shape[0], result.shape[1], amount_of_features))  
    
    print(seq_array.shape)
        
    try:
        prediction = loaded_model.predict(seq_array)
        pred = prediction.tolist()
        return(pred)
    except Exception as e:
        return(str(e))
    
if __name__ == "__main__":
    init()
    run('[{"Open":0.7127071619,"High":0.7314814925,"Low":0.5424528122,"Volume":1.0,"Close":0.9403669834},{"Open":0.9337016344,"High":1.0,"Low":1.0,"Volume":0.4399125874,"Close":0.8715596199},{"Open":0.9226519465,"High":0.879629612,"Low":0.9858490825,"Volume":0.1769355834,"Close":1.0},{"Open":1.0,"High":0.805555582,"Low":0.6556603909,"Volume":0.354439944,"Close":0.770642221},{"Open":0.8674033284,"High":0.662037015,"Low":0.5613207817,"Volume":0.3036391139,"Close":0.43577981},{"Open":0.2486187816,"High":0.3240740597,"Low":0.4386792481,"Volume":0.1165050864,"Close":0.5275229216},{"Open":0.3038673997,"High":0.2037037015,"Low":0.0,"Volume":0.4684149027,"Close":0.0},{"Open":0.0718232021,"High":0.0,"Low":0.0330188684,"Volume":0.2322592139,"Close":0.0},{"Open":0.0055248621,"High":0.1666666716,"Low":0.1179245263,"Volume":0.2997646928,"Close":0.2706421912},{"Open":0.0,"High":0.0740740746,"Low":0.1839622706,"Volume":0.0,"Close":0.2798165083},{"Open":0.2209944725,"High":0.3356481493,"Low":0.4528301954,"Volume":0.2173066139,"Close":0.56422019}]')