import numpy as np
import json
from typing import TextIO
from sklearn.preprocessing import StandardScaler()

def create_data(num_points):
  x_train = np.random.rand(num_points, 3)
  y_train = np.exp(x_train[:, 0]*x_train[:, 1]-3*x_train[:, 2]+1) + 3*x_train[:, 1]**4 + 0.5*np.random.rand(num_points)
  x_test = np.random.rand(np.int(num_points/4), 3)
  y_test = np.exp(x_test[:, 0]*x_test[:, 1]-3*x_test[:, 2]+1) + 3*x_test[:, 1]**4 + 0.5*np.random.rand(np.int(num_points/4))
  
  data_dict = {}
  data_dict['X_train'] = x_train.tolist()
  data_dict['X_test'] = x_test.tolist()
  data_dict['y_train'] = y_train.tolist()
  data_dict['y_test'] = y_test.tolist()

  return data_dict

def preprocess_data(data):
  x_train = np.array(data['X_train'])
  x_test = np.array(data['X_test'])
  y_train = np.array(data['y_train'])
  y_test = np.array(data['y_test'])
  
  x_train_std = StandardScaler().fit_transform(x_train)
  y_train_std = StandarScaler().fit_transform(y_train)
  x_test_std = StandardScaler().fit(x_train).transform(x_test)
  y_test_std = StandardScaler().fit(y_train).transform(y_test)
  
  preprocess_data_dict = {}
  preprocess_data_dict['X_train'] = x_train_std
  preprocess_data_dict['X_test'] = x_test_std
  preprocess_data_dict['y_train'] = y_train_std
  preprocess_data_dict['y_test'] = y_test_std
  
  return preprocess_data_dict
  
def save_data(datas:list, filenames:list) -> None:
    """
    Saves data as JSON.
    Args:
      datas (list):
        A list of dicts of data to save.
      filenames (list):
        A list of filenames corresponding to the data dicts to save the data in. 
        These should have a '.json' extension.
    """

    for i in range(len(datas)):
        data = datas[i]
        filename = filenames[i]

        try:
            data["schema"] = "data"
        except KeyError as e:
            print(f'Error: Could not load schema key from {filename}')

        try:
            with open(filename,'w') as f:
                # Write data to file as this will serve as output artifact
                f.write(json.dumps(data, indent=2)) 
        except IOError as e:
            print(f'Error: Could not open {filename}')
            
def load_data(filename:TextIO) -> dict:
    """
    Loads data from JSON.
    Args:
      filename (TextIO):
        The file to load the data from.
    Returns:
      data (dict):
        The data that was loaded from the file.
    """

    if isinstance(filename, str):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

        except IOError:
            print(f'Error: Could not open {filename}')

    else:
        data = json.load(filename)

    return data
