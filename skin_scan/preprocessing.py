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
import joblib

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
    save_preprocessor_to_gcs(preprocessor)
    return array

def run_X_pipeline(df: pd.DataFrame):
    preprocessor = load_preprocessor_from_gcs()
    data = preprocessor.transform(df)
    return data

BUCKET_NAME = "skin_scan_mohnatz"
CLASS_NAMES_PATH = "models/class_names.joblib"
LOCAL_REGISTRY_PATH = "preprocessing_pipeline"

def save_class_names_to_gcs(class_names):
    # Ensure local registry exists
    os.makedirs(LOCAL_REGISTRY_PATH, exist_ok=True)
    local_path = os.path.join(LOCAL_REGISTRY_PATH, os.path.basename(CLASS_NAMES_PATH))

    # Save locally
    joblib.dump(class_names, local_path)

    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(CLASS_NAMES_PATH)
    blob.upload_from_filename(local_path)

    print(f"✅ class_names saved to GCS at gs://{BUCKET_NAME}/{CLASS_NAMES_PATH}")

def load_class_names_from_gcs():
    local_path = os.path.join(LOCAL_REGISTRY_PATH, os.path.basename(CLASS_NAMES_PATH))

    # Download from GCS
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(CLASS_NAMES_PATH)
    blob.download_to_filename(local_path)

    # Load the class names
    class_names = joblib.load(local_path)
    print(f"✅ class_names loaded from GCS")
    return class_names

def run_y_pipeline(df: pd.DataFrame) -> np.array:
    '''Processes the y dataframe so that all the values are Numeric and model ready'''
    y_pipeline = Pipeline([('ohe', OneHotEncoder(sparse_output=False, drop=None))])
    y_encoded = y_pipeline.fit_transform(df)
    class_names = y_pipeline.named_steps['ohe'].categories_[0]
    save_class_names_to_gcs(class_names)
    return y_encoded

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
    y, class_names = run_y_pipeline(y)
    X_metadata = processed_metadata.drop(columns=[col for col in ['dx_type','lesion_id',"dx","resized_image","image_id"]
                                       if col in processed_metadata.columns])
    X_metadata = run_X_pipeline(X_metadata)
    return X_metadata, y, class_names


# Constants
BUCKET_NAME = "skin_scan_mohnatz"
BLOB_PATH = "models/preprocessor_joblib"
LOCAL_REGISTRY_PATH = "preprocessing_pipeline"

def save_preprocessor_to_gcs(preprocessor: Pipeline):
    """
    Save a scikit-learn preprocessor locally and upload it to GCS.
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # Ensure local directory exists
    os.makedirs(LOCAL_REGISTRY_PATH, exist_ok=True)
    local_path = os.path.join(LOCAL_REGISTRY_PATH, os.path.basename(BLOB_PATH))

    try:
        # Save model locally (just once!)
        joblib.dump(preprocessor, local_path)
        print(f"✅ Preprocessor saved locally at {local_path}")

        # Upload to GCS
        blob = bucket.blob(BLOB_PATH)
        blob.upload_from_filename(local_path)
        print(f"✅ Preprocessor uploaded to GCS at gs://{BUCKET_NAME}/{BLOB_PATH}")

    except Exception as e:
        print(f"❌ Failed to save model to GCS: {e}")

def load_preprocessor_from_gcs() -> Pipeline:
    """
    Download a scikit-learn preprocessor from GCS and load it.

    Returns:
        The loaded preprocessor (Pipeline or ColumnTransformer), or None if failed.
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(BLOB_PATH)

    # Ensure local directory exists
    os.makedirs(LOCAL_REGISTRY_PATH, exist_ok=True)
    local_path = os.path.join(LOCAL_REGISTRY_PATH, os.path.basename(BLOB_PATH))

    try:
        # Download the file from GCS
        blob.download_to_filename(local_path)
        print(f"✅ Preprocessor downloaded from GCS to {local_path}")

        # Load the preprocessor
        preprocessor = joblib.load(local_path)
        print("✅ Preprocessor successfully loaded from local file")
        return preprocessor

    except Exception as e:
        print(f"❌ Failed to load preprocessor from GCS: {e}")
        return None