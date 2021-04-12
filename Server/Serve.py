from fastapi import FastAPI, Response
import numpy as np
from pydantic import BaseModel
import json
import time
import sys
import mlflow
from utils import *

app = FastAPI()
obj = DriveAPI()


logged_model = './../src_image/mlruns/0/fb9fee4e492b4dbaad0cf10106e60151/artifacts/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

class Item(BaseModel):
    images: list

class Item_uri(BaseModel):
    name: str

@app.get('/')
async def index():
    return "hello word"

@app.post('/predict')
async def predict(item: Item):
    images = np.array(item.images)
    res = loaded_model.predict(images).tolist()
    return res

@app.post('/update')
async def model_update(item: Item_uri):
    f_name = item.name
    obj.FileDownload(f_name)

    dezip(f_name, 'model')
    loaded_model = mlflow.pyfunc.load_model('model')
    return "model updated"

