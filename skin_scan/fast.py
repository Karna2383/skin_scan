import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
import iphone_input_img_proc as iphone_proc
from model import load_model_from_gcs
import preprocessing
import data
import numpy as np

app = FastAPI()

@app.get("/")
def root():
    return {'greeting': 'Hello'}


@app.get("/predict")
def predict(image, X_pred):      # 1
    # converting this image into numpy array
    image = iphone_proc.process_input_image(image)
    X_image = np.expand_dims(image / 255, axis=0)  # shape becomes (1, 96, 96, 3)

    X_metadata = preprocessing.run_X_pipeline(X_pred)

    model = load_model_from_gcs()
    result = model.predict([X_image,X_metadata])


    predicted_probs = result[0]  # since result has shape (1, 7), take the first row
    # class_prob_dict = dict(zip(class_names, predicted_probs))
    # sorted_result = dict(sorted(class_prob_dict.items(), key=lambda x: x[1], reverse=True))
    # pretty_result = {k: round(float(v), 3) for k, v in sorted_result.items()}
    return result
