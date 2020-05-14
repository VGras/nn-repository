import numpy as np
import json

def create_data():
  x_train = np.random.rand(1000, 3)
  y_train = np.exp(x_train[:, 0]*x_train[:, 1]-3*x_train[:, 2]+1) + 3*x_train[:, 1]**4 + 0.5*np.random.rand(1000)
  x_test = np.random.rand(300,3)
  y_test = np.exp(x_test[:, 0]*x_test[:, 1]-3*x_test[:, 2]+1) + 3*x_test[:, 1]**4 + 0.5*np.random.rand(300)
  
  data_dict = {}
  data_dict['X_train'] = x_train.tolist()
  data_dict['X_test'] = x_test.tolist()
  data_dict['y_train'] = y_train.tolist()
  data_dict['y_test'] = y_test.tolist()
  data_dict['schema'] = 'data'
  
  with open('data.json','w') as f:
      # Write data to file as this will serve as output artifact
      f.write(json.dumps(data_dict, indent=2)) 

  
