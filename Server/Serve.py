from fastapi import FastAPI, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
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
        self.preprocess = None

class List_Models():
    def __init__(self):
        self.models = {}

class List_Pipelines():
    def __init__(self):
        self.pipelines = {}

class Pipeline():
    def __init__(self):
        self.graph = {}
    
    def parcours(self, foo=lambda x: x):
        self.graph = self._parcours(self.graph, foo)
    
    def _parcours(self,elt, foo=lambda x: x):
        print(elt["name"])
        elt = foo(elt)
        if "children" in elt:
            for i, child in enumerate(elt["children"]):
                elt["children"][i] = self._parcours(child, foo)
        return elt


# Initialisation
app = FastAPI()
obj = DriveAPI()

# init model
list_models = List_Models()
list_pipelines = List_Pipelines()

# requets
class Item(BaseModel):
    data: list


class Item_uri(BaseModel):
    name: str
    version: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
async def index():
    return "Server Up"


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
    
    list_models.models[name].model = mlflow.pyfunc.load_model(name+'/model')
    try:
        list_models.models[name].preprocess = pickle.load(open(name+'/preprocessing.pkl','r'))
    except:
        print("no preprocessing")
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
        filehandler = open('Data/image/'+filename, 'w') # wtf => a mettre dans le nom du truc ou pipeline
        pickle.dump(data, filehandler)
    else:
        filehandler = open('Data/text/'+filename, 'w')
        pickle.dump(data, filehandler)
    res=None
    if name in list_models.models:
        processed_data = list_models.models[name].preprocess.preprocess(data)
        res = list_models.models[name].model.predict(processed_data).tolist()
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

        list_models.models[name].model = mlflow.pyfunc.load_model(name+'/model')
        try:
            list_models.models[name].preprocess = pickle.load(open(name+'/preprocessing.pkl','r'))
        except:
            print("no preprocessing")
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

@app.post('/pipeline/set/{name}')
async def set_pipe(name, file: UploadFile = File(...)):
    contents = await file.read()
    new_pipe = Pipeline()
    js = contents.decode('utf8').replace("'", '"')
    new_pipe.graph = json.loads(js)
    list_pipelines.pipelines[name] = new_pipe

    def add_xy(elt): #pour graph react
        elt["textProps"] = {"x": -25, "y": 25}
        return elt
    
    new_pipe.parcours(add_xy)

    def add_info(elt):
        if elt["name"] in list_models.models:
            elt["version"] = list_models.models[elt["name"]].version
        return elt

    new_pipe.parcours(add_info)

    return "new pipepile set"

@app.get('/pipeline/{name}')
async def get_pipe(name):
    if name in list_pipelines.pipelines:
        return list_pipelines.pipelines[name].graph
    return None