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

model_image = Model()

model_text = Model()

class Item(BaseModel):
    images: list

class Item_uri(BaseModel):
    name: str
    version: str
    type: str

@app.get('/')
async def index():
    return "hello word"


        

@app.post('/update')
async def model_update(item: Item_uri):
    f_name = item.name
    type = item.type
    meta_data = dict(
        last_f_name = f_name,
        last_version = item.version
    )
    if type == 'image':
        with open('meta_data_image.yml', 'w') as outfile:
            yaml.dump(meta_data, outfile, default_flow_style=False)
        model_image.version = item.version
        obj.FileDownload(f_name)
        dezip(f_name, 'model_image')
        os.remove(f_name)
        model_image.model = mlflow.pyfunc.load_model('model_image')
    else:
        with open('meta_data_text.yml', 'w') as outfile:
            yaml.dump(meta_data, outfile, default_flow_style=False)
        model_text.version = item.version
        obj.FileDownload(f_name)
        dezip(f_name, 'model_text')
        os.remove(f_name)
        model_text.model = mlflow.pyfunc.load_model('model_text')


    return "model updated"


@app.post('/predictImageClassification')
async def predict(item: Item):
    if model_image.version:
        print("model found")
        images = np.array(item.images)
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
        images = np.array(item.images)
        res = model_image.model.predict(images).tolist()

        return res

@app.post('/predictTextClassification')
async def predict(item: Item):
    if model.version:
        print("model found")
        textes = np.array(item.text)
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
        textes = np.array(item.text)
        res = model_text.model.predict(textes).tolist()

        return res
        


@app.get('/versionTextClassification')
async def get_hash():
    return model_text.version

@app.get('/versionImageClassification')
async def get_hash():
    return model_image.version