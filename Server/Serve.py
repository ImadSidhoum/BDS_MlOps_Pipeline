from fastapi import FastAPI, Response
import numpy as np
from pydantic import BaseModel
import json
import time
import sys
import mlflow
from utils import *
import os ,yaml

app = FastAPI()
obj = DriveAPI()


class Model():
    def __init__(self):
        self.version = None
        self.model = None

model = Model()

class Item(BaseModel):
    images: list

class Item_uri(BaseModel):
    name: str
    version: str

@app.get('/')
async def index():
    return "hello word"

@app.post('/predict')
async def predict(item: Item):
    if model.version:
        print("model found")
        images = np.array(item.images)
        res = model.model.predict(images).tolist()
        return res
    else:
        print("no model found !! loadig the last model")
        # Read YAML file
        with open("meta_data.yml", 'r') as infile:
            meta_data = yaml.load(infile)
        f_name = meta_data['last_f_name']
        obj.FileDownload(f_name)
        dezip(f_name, 'model')
        os.remove(f_name)
        model.model = mlflow.pyfunc.load_model('model')
        model.version = meta_data['last_version']

        # Predicting
        images = np.array(item.images)
        res = model.model.predict(images).tolist()

        return res
        

@app.post('/update')
async def model_update(item: Item_uri):
    f_name = item.name
    model.version = item.version
    meta_data = dict(
        last_f_name = f_name,
        last_version = model.version
    )
    with open('meta_data.yml', 'w') as outfile:
        yaml.dump(meta_data, outfile, default_flow_style=False)
    obj.FileDownload(f_name)
    dezip(f_name, 'model')
    os.remove(f_name)
    model.model = mlflow.pyfunc.load_model('model')
    return "model updated"

@app.get('/version')
async def get_hash():
    return model.version
