import pandas as pd
#Pipline Imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from PIL import Image
import io
import os
import numpy as np
import joblib
import json
from io import BytesIO
from google.cloud import storage

def process_input_image(image: Image.Image) -> np.ndarray:
    """
    Resizes and normalizes a skin lesion PIL image to (96, 96, 3).
    """
    resized_image = image.resize((96, 96))
    image_array = np.array(resized_image).astype("float32") / 255.0
    return image_array

def create_X_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    age_pipeline = Pipeline([('scaler', MinMaxScaler())])
    cat_pipeline = Pipeline([('ohe', OneHotEncoder(sparse_output=False, drop='first'))])
    preprocessor = ColumnTransformer([
    ( 'age', age_pipeline, ['age']),
    ('cat', cat_pipeline,['sex','localization'])
    ])
    array = preprocessor.fit_transform(df)
    return array

BUCKET_NAME = "skin_scan_mohnatz"
LOCAL_REGISTRY_PATH = "preprocessing_pipeline"

def run_X_pipeline(df: pd.DataFrame):
    preprocessor = load_preprocessor_local()
    data = preprocessor.transform(df)
    return data

def run_y_pipeline(df: pd.DataFrame) -> np.array:
    '''Processes the y dataframe so that all the values are Numeric and model ready'''
    y_pipeline = Pipeline([('ohe', OneHotEncoder(sparse_output=False, drop=None))])
    y_encoded = y_pipeline.fit_transform(df)
    return y_encoded


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
    return df


def prepare_data_for_model(processed_metadata: pd.DataFrame) -> tuple[pd.DataFrame, np.array, np.array]:
    y = processed_metadata[["dx"]]
    y = run_y_pipeline(y)
    X_metadata = processed_metadata.drop(columns=[col for col in ['dx_type','lesion_id',"dx","resized_image","image_id"]
                                       if col in processed_metadata.columns])
    X_metadata = run_X_pipeline(X_metadata)
    return X_metadata, y


def load_preprocessor_local() -> Pipeline:
    """
    Load a scikit-learn preprocessor from a local file.

    Returns:
        The loaded preprocessor (Pipeline or ColumnTransformer), or None if failed.
    """
    current_dir = os.path.dirname(__file__)
    rel_path = os.path.join(current_dir, "..", "preprocessing_pipeline", "preprocessor_joblib")
    abs_path = os.path.abspath(rel_path)

    try:
        preprocessor = joblib.load(abs_path)
        print(f"✅ Preprocessor successfully loaded from {abs_path}")
        return preprocessor
    except Exception as e:
        print(f"❌ Failed to load preprocessor: {e}")
        return None
