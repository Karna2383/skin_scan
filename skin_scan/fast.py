import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

@app.get("/")
def root():
    return {'greeting': 'Hello'}


@app.get("/predict")
def predict(
        image
    ):      # 1
    # converting this image into numpy array

    # preprocess this image

    #load the model

    #model.predict -> dict of 7 classes as key: value are probability)
    result =  {'nv': 0.77,
               'mel': 0.23}
    return result
