import os
import sys
import yaml
import numpy as np
import pandas as pd
from exception import HousePriceException

def load_csv(file_path):

    try:
        return pd.read_csv(file_path)

    except Exception as e:
            raise HousePriceException(e, sys) from e

def save_csv(data,path):

    try:
        data.to_csv(path,index=False)

    except Exception as e:
        raise HousePriceException(e, sys) from e

def read_yaml(filepath:str):

    try:
        with open(filepath,"r") as file:
            return yaml.safe_load(file)

    except Exception as e:
        raise HousePriceException(e, sys) from e

def save_numpy_array_data(array: np.array,file_path):

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise HousePriceException(e, sys) from e