from fastapi import FastAPI, Response
import numpy as np
from pydantic import BaseModel
import json
import time
import sys
import mlflow

app = FastAPI()

logged_model = './../../BDS_MlOps_Pipeline/src_image/mlruns/0/fb9fee4e492b4dbaad0cf10106e60151/artifacts/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

class Item(BaseModel):
    images: list

class Item_uri(BaseModel):
    uri: str

@app.get('/')
async def index():
    return "hello word"

@app.post('/predict')
async def predict(item: Item):
    images = np.array(item.images)
    res = loaded_model.predict(images).tolist()
    return res

@app.post('/model')
async def model_update(item: Item_uri):
    uri = item.uri
    loaded_model = mlflow.pyfunc.load_model(uri)