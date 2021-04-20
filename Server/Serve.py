from fastapi import FastAPI, Response
import numpy as np
from pydantic import BaseModel
import json
import time
import sys
import mlflow
sys.path.insert(0, '..')
from utils import *
import os ,yaml

class Model():
    def __init__(self):
        self.version = None
        self.model = None

class List_Models():
    def __init__(self):
        self.models = {}

# Initialisation
app = FastAPI()
obj = DriveAPI()

# init model
list_models = List_Models()

# requets
class Item(BaseModel):
<<<<<<< HEAD
    data: list
=======
    items: list
>>>>>>> 492c82c1d285d96a47320c14e35d96c3c6d54369

class Item_uri(BaseModel):
    name: str
    version: str

@app.get('/')
async def index():
    return "Server Up"

#@app.get('/yaml')
#async def get_yaml():
#with open('meta_data_image.yml', 'w') as outfile:
#            yaml.dump(meta_data, outfile, default_flow_style=False)



<<<<<<< HEAD
@app.post('/update/{name}')
async def model_update(item: Item_uri, name):
    f_name = item.name
    obj.FileDownload(f_name)
    dezip(f_name, name)
    os.remove(f_name)

    if name in list_models.models:
        print(f"model {name} found")
    else:
        list_models.models[name] = Model()
    
    list_models.models[name].model = mlflow.pyfunc.load_model(name)
    list_models.models[name].version = item.version
    return "model updated"
=======
    return "model updated"


@app.post('/predictImageClassification')
async def predict(item: Item):
    if model_image.version:
        print("model found")
        images = np.array(item.items)
        res = model_image.model.predict(images).tolist()
        return res
    else:
        print("no model found !! loadig the last model")
        # Read YAML file
        with open("meta_data_image.yml", 'r') as infile:
            meta_data = yaml.load(infile)
        f_name = meta_data['last_f_name']
        obj.FileDownload(f_name)
        dezip(f_name, 'model_image')
        os.remove(f_name)
        model_image.model = mlflow.pyfunc.load_model('model_image')
        model_image.version = meta_data['last_version']

        # Predicting
        images = np.array(item.items)
        res = model_image.model.predict(images).tolist()

        return res

@app.post('/predictTextClassification')
async def predict(item: Item):
    if model_text.version:
        print("model found")
        textes = np.array(item.items)
        res = model_text.model.predict(textes).tolist()
        return res
    else:
        print("no model found !! loadig the last model")
        # Read YAML file
        with open("meta_data_text.yml", 'r') as infile:
            meta_data = yaml.load(infile)
        f_name = meta_data['last_f_name']
        obj.FileDownload(f_name)
        dezip(f_name, 'model_text')
        os.remove(f_name)
        model_text.model = mlflow.pyfunc.load_model('model_text')
        model_text.version = meta_data['last_version']

        # Predicting
        textes = np.array(item.items)
        res = model_text.model.predict(textes).tolist()

        return res
        


@app.get('/versionTextClassification')
async def get_hash():
    return model_text.version
>>>>>>> 492c82c1d285d96a47320c14e35d96c3c6d54369

@app.post('/predict/{name}')
async def predict(item: Item, name):
    data = np.array(item.data)
    res=None
    if name in list_models.models:
        res = list_models.models[name].model.predict(data).tolist()
    return res
    
@app.get('/version/{name}')
async def get_hash(name):
    if name in list_models.models:
        return list_models.models[name].version
    return None


if len(sys.argv) == 2:
    with open(sys.argv[-1], 'r') as infile:
        meta_data = yaml.load(infile)
    f_name = meta_data['last_f_name']
    obj.FileDownload(f_name)
    dezip(f_name, 'model_image')
    os.remove(f_name)
    model_image.model = mlflow.pyfunc.load_model('model_image')
    model_image.version = meta_data['last_version']