import json
import numpy as np
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, model_from_json, load_model, save_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import callbacks

def build_model(data, layers=[50, 50], dropout=[0.1, 0.1], activations=['tanh', 'tanh']):
    
    x_train = np.array(data['X_train'])
    y_train = np.array(data['y_train'])
    input_shape = (x_train.shape[1],)
    try:
        output_shape = y_train.shape[1]
    except IndexError:
        output_shape = 1
    
    model = Sequential()
    for i in range(len(layers)):
        model.add(Dense(layers[i], activation=activations[i], input_shape=input_shape))
        model.add(Dropout(dropout[i]))
    model.add(Dense(output_shape, activation='linear'))

    return model
    
    
def train_model(model, data, epochs=5, batch_size=10):
    
    x_train = np.array(data['X_train'])
    y_train = np.array(data['y_train'])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['MeanSquaredError'])
    fithistory = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    return fithistory.history, model
    
    
def test_model(model, data):
    
    x_test = np.array(data['X_test'])
    y_test = np.array(data['y_test'])

    score = model.evaluate(x_test, y_test, verbose=0)
    res = {}
    res['schema'] = 'score_nn'
    res['Test loss'] = score[0]
    res['Test accuracy'] = score[1]
    res['Test r2 score'] = r2_score(y_test, model.predict(x_test))
    
    with open("results.json",'w') as f:
        f.write(json.dumps(res))
        
        
def save_model_json(model:Sequential, filename:str) -> None:
    """
    Saves a model's architecture and weights as a JSON file. The output JSON will 
    contain a "model" field with "specs" and "weights" fields inside it. The 
    "specs" field contains the model's architecture and the "weights" field
    contains the weights.
    Args:
      model (keras.models.Sequential):
        The model to save.
      filename (str):
        The name of the file to save the model in. This should have a '.json'
        extension.
    """

    model_dict = {"model":{}}

    model_json = model.to_json()
    model_dict["model"]["specs"] = json.loads(model_json)

    weights = model.get_weights()
    # Convert weight arrays to lists because those are JSON compatible
    weights = nested_arrays_to_lists(weights)
    model_dict["model"]["weights"] = weights

    model_dict["schema"] = "orquestra-v1-model"

    try:
        with open(filename, "w") as f:
            f.write(json.dumps(model_dict, indent=2))
    except IOError:
        print('Error: Could not load {filename}')


def load_model_json(filename:str) -> Sequential:
    """
    Loads a keras model from a JSON file.
    Args:
      filename (str):
        The JSON file to load the model from. This file must contain a "model" 
        field with "specs" and "weights" fields inside it. The "specs" field 
        should contain the model's architecture and the "weights" field should
        contain the weights. (The format saved by the `save_model_json` function.)
    Returns:
      model (Sequential):
        The model loaded from the file. 
    """

    # load json and create model
    with open(filename) as json_file:
        loaded_model_artifact = json.load(json_file)

    loaded_model = json.dumps(loaded_model_artifact["model"]["specs"])
    loaded_model = model_from_json(loaded_model)

    try:
        weights = loaded_model_artifact["model"]["weights"]
    except KeyError:
        print(f'Error: Could not load weights from {weights}')

    # Everything below the top-level list needs to be converted to a numpy array
    # because those are the types expected by `set_weights`
    for i in range(len(weights)):
        weights[i] = nested_lists_to_arrays(weights[i])
    loaded_model.set_weights(weights)

    return loaded_model


def nested_arrays_to_lists(obj):
    """
    Helper function for saving models in JSON format. Converts nested numpy 
    arrays to lists (lists are JSON compatible).
    """

    if isinstance(obj, np.ndarray):
        obj = obj.tolist()

    try:
        for i in range(len(obj)):
            obj[i] = nested_arrays_to_lists(obj[i])

    except TypeError:
        return obj

    return obj

def nested_lists_to_arrays(obj):
    """
    Helper function for loading models in JSON format. Converts nested lists to numpy arrays (numpy arrays are the expected type to set the weights 
    of a Keras model).
    """

    if isinstance(obj, list):
        obj = np.array(obj)

    try:
        for i in range(len(obj)):
            obj[i] = nested_lists_to_arrays(obj[i])
    except TypeError:
        return obj

    return obj


def save_model_h5(model:Sequential, filename:str) -> None:
    """
    Saves a complete model as an H5 file. H5 files can be used to pass models 
    between tasks but cannot be returned in a workflowresult.
    Args:
      model (keras.models.Sequential):
        The model to save.
      filename (str):
        The name of the file to save the model in. This should have a '.h5'
        extension.
    """
    keras.models.save_model(
        model, filename, include_optimizer=True
    )


def load_model_h5(filename:str) -> Sequential:
    """
    Loads a keras model from an H5 file. H5 files can be used to pass models 
    between tasks but cannot be returned in a workflowresult.
    Args:
      filename (str):
        The H5 file to load the model from. This should have the format created 
        by `keras.models.save_model`.
    Returns:
      model (Sequential):
        The model loaded from the file. 
    """

    model = keras.models.load_model(filename, compile=True)
    return model

def save_loss_history(history, filename:str) -> None:
    """
    Saves a keras.History.history object to a JSON file.
    Args:
      history (keras.History.history):
        The history object to save.
      filename (str):
        The name of the file to save the history in. This should have a '.json'
        extension.
    """

    history_dict = {}
    history_dict["history"] = history
    history_dict["schema"] = "orquestra-v1-loss-function-history"

    try:
        with open(filename, "w") as f:
            f.write(json.dumps(history_dict, indent=2))
    except IOError:
        print(f'Could not write to {filename}')
