from fastapi import FastAPI, Response, UploadFile, File
import numpy as np
from pydantic import BaseModel
import json
import time
import sys
import mlflow
sys.path.insert(0, '..')
from atosflow.utils import *
import os ,yaml
import pickle 


class Model():
    def __init__(self):
        self.version = None
        self.model = None
        self.name = None

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
    data: list


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



@app.post('/update/{name}')
async def model_update(item: Item_uri, name):
    
    if name in list_models.models:
        os.remove(name)
    else:
        list_models.models[name] = Model()

    f_name = item.name
    obj.FileDownload(f_name)
    dezip(f_name, name)
    os.remove(f_name)
    
    list_models.models[name].model = mlflow.pyfunc.load_model(name)
    list_models.models[name].version = item.version
    list_models.models[name].name = f_name
    return "model updated"

@app.post('/predict/{name}')
async def predict(item: Item, name):
    data = np.array(item.data)
    date = datetime.datetime.now()
    filename = str(date.strftime("%m-%d-%y_%X"))
    filename = filename.replace(":","-")
    if name =='image':
        filehandler = open('Data/image/'+filename, 'w')
        pickle.dump(data, filehandler)
    else:
        filehandler = open('Data/text/'+filename, 'w')
        pickle.dump(data, filehandler)
    res=None
    if name in list_models.models:
        res = list_models.models[name].model.predict(data).tolist()
    return res
    
@app.get('/version/{name}')
async def get_hash(name):
    if name in list_models.models:
        return list_models.models[name].version
    return None

@app.post('/config/set')
async def set_yaml(file: UploadFile = File(...)):
    contents = await file.read()
    meta_data = yaml.load(contents)
    print(meta_data)
    for name in meta_data.keys():
        if name in list_models.models:
            os.remove(name)
        else:
            list_models.models[name] = Model()

        f_name = meta_data[name]["name"]
        obj.FileDownload(f_name)
        dezip(f_name, name)
        os.remove(f_name)

        list_models.models[name].model = mlflow.pyfunc.load_model(name)
        list_models.models[name].version =meta_data[name]["version"]
        list_models.models[name].name = f_name
    return "new config set"



@app.get('/config/get')
async def get_yaml():
    meta_data = {}
    for elt in list_models.models.keys():
        tmp = {}
        tmp["name"] = list_models.models[elt].name
        tmp["version"] =list_models.models[elt].version
        meta_data[elt] = tmp
    
    with open('config.yml', 'w') as outfile:
        yaml.dump(meta_data, outfile, default_flow_style=False)
    
    f = open("config.yml", "r")
    return f.read()

