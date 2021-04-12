from fastapi import FastAPI, Response
import numpy as np
from pydantic import BaseModel
import json
import time
import sys
import mlflow
from utils import *
import os


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
        images = np.array(item.images)
        res = model.model.predict(images).tolist()
        return res
    return "no model"

@app.post('/update')
async def model_update(item: Item_uri):
    f_name = item.name
    model.version = item.version

    obj.FileDownload(f_name)
    dezip(f_name, 'model')
    os.remove(f_name)
    model.model = mlflow.pyfunc.load_model('model')
    return "model updated"

@app.get('/version')
async def get_hash():
    return model.version
