import os
import sys
import yaml
import json
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
        with open(filepath,"rb") as file:
            return yaml.safe_load(file)

    except Exception as e:
        raise HousePriceException(e, sys) from e

def load_numpy_array_data(file_path):

    try:
         with open(file_path, "rb") as file_obj:
            return np.load(file_obj)

    except Exception as e:
        raise HousePriceException(e, sys) from e

def save_numpy_array_data(array: np.array,file_path):

    try:
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise HousePriceException(e, sys) from e

def save_json(data,file_path):

    try:
        with open(file_path, 'w') as file:
            json.dump(data,file)

    except Exception as e:
        HousePriceException(e,sys)