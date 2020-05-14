import numpy as np
import json

def create_data():
  x_train = np.random.rand(100, 3)
  y_train = np.exp(X_train[0]*X_train[1]-3*X_train[2]+1) + 3* X_train[1]**4 + 0.5*np.random.rand(100)
  x_test = np.random.rand(30,3)
  y_test = np.exp(X_test[0]*X_test[1]-3*X_test[2]+1) + 3* X_test[1]**4 + 0.5*np.random.rand(100)
  
  data_dict = {}
  data_dict['X_train'] = x_train.tolist()
  data_dict['X_test'] = x_test.tolist()
  data_dict['y_train'] = y_train.tolist()
  data_dict['y_test'] = y_test.tolist()
  data['schema'] = 'data'
  
  with open(data.json,'w') as f:
      # Write data to file as this will serve as output artifact
      f.write(json.dumps(data, indent=2)) 

  
