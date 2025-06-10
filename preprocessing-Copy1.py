import pandas as pd
#Pipline Imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from google.cloud import storage
from PIL import Image
import io
import os
import numpy as np

def run_X_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    age_pipeline = Pipeline([('scaler', MinMaxScaler())])
    cat_pipeline = Pipeline([('ohe', OneHotEncoder(sparse_output=False, drop='first'))])
    preprocessor = ColumnTransformer([
    ( 'age', age_pipeline, ['age']),
    ('cat', cat_pipeline,['sex','localization'])
    ])
    array = preprocessor.fit_transform(df)
    return array

def run_y_pipeline(df: pd.DataFrame) -> np.array:
    '''Processes the y dataframe so that all the values are Numeric and model ready'''
    y_pipeline = Pipeline([('ohe', OneHotEncoder(sparse_output=False, drop=None))])
    df = y_pipeline.fit_transform(df)
    return df

def preprocess_images(width:int, height:int, bucket_name="skin_scan_mohnatz") -> pd.DataFrame:
    """Retrieves the images from the bucket and
    returns a dataframe with image_id and a numpy array of selected shapes (height, width, 3)"""
    print("Hello")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix="train_all_images/")
    images = [blob.name for blob in blobs if blob.name.lower().endswith(".jpg")]
    image_name = []
    resized_array = []
    for index, image in enumerate(images):
        blob = bucket.blob(image)
        image_bytes = blob.download_as_bytes()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((width, height))
        resized_array.append(np.array(img))
        image_name.append(os.path.basename(image).split('.')[0])
    output_df = pd.DataFrame({
    'image_id': image_name,
    'resized_array': resized_array
})
    return output_df


def preprocess_metadata(df: pd.DataFrame, split=True):# -> tuple[pd.DataFrame, pd.DataFrame]:
    '''preprocess's metadata from the skin-cancer-mnist-ham10000 dataset
    returns a preprocessed version of X and y'''
    df = df.drop(columns=[col for col in ['dx_type','lesion_id'] if col in df.columns])
    # fill age with mean values
    df['age'] = df['age'].fillna((df['age'].mean()))
    # Drop the unknown sex names
    df = df[df['sex'] != 'unknown']
    #drop unknowns
    df = df[df['localization'] != 'unknown']
    df = df.sort_values(by="image_id")
    # return processed df
    # TODO put the image processing part here, this should separate out of metadata and get the array here!
    return df


def prepare_data_for_model(processed_metadata: pd.DataFrame) -> tuple[pd.DataFrame, np.array, np.array]:
    y = processed_metadata[["dx"]]
    y = run_y_pipeline(y)
    X_metadata = processed_metadata.drop(columns=[col for col in ['dx_type','lesion_id',"dx","resized_image","image_id"]
                                       if col in processed_metadata.columns])
    X_metadata = run_X_pipeline(X_metadata)
    return X_metadata, y
