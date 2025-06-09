import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, concatenate
import os
from google.cloud import storage
from tensorflow import keras

def create_model() -> Model:
    # Image Branch
    image_input = Input(shape=(96, 96, 3))
    cnn = Conv2D(16, (3, 3), activation='relu')(image_input)
    cnn = MaxPooling2D(pool_size=(2, 2))(cnn)
    cnn = MaxPooling2D(2, 2)(cnn)
    cnn = Conv2D(32, (3, 3), activation='relu')(cnn)
    cnn = MaxPooling2D(2, 2)(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(64, activation='relu')(cnn)

    # Metadata Branch
    meta_input = Input(shape=(15,))
    dln = Dense(32, activation='relu')(meta_input)
    dln = Dense(16, activation='relu')(dln)

    # Combined Final Layers
    combined_layers = concatenate([dln, meta_input])
    HAM_combined_model = Dense(32)(combined_layers)
    HAM_combined_model = Dense(7, activation='softmax')(HAM_combined_model)

    return Model(inputs=[image_input, meta_input], outputs=HAM_combined_model)

def fit_model(X_images: np.array, X_metadata: np.array, y: np.array) -> Model:
    model = create_model()
    model.fit([X_images, X_metadata], y, epochs=20, batch_size=32, validation_split=0.2)
    save_model_to_gcs(model)

def predict(X_images: np.array, X_metadata: np.array):
    model = load_model_from_gcs()
    prediction = model.predict([X_images, X_metadata])
    return prediction

# Constants
BUCKET_NAME = "skin_scan_mohnatz"
BLOB_PATH = "models/96_96_metadata_friday_model.keras"
LOCAL_REGISTRY_PATH = "model"  # or any local folder you prefer

def load_model_from_gcs():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(BLOB_PATH)

    # Make sure local folder exists
    os.makedirs(LOCAL_REGISTRY_PATH, exist_ok=True)
    local_path = os.path.join(LOCAL_REGISTRY_PATH, os.path.basename(BLOB_PATH))

    try:
        # Download the model from GCS
        blob.download_to_filename(local_path)

        # Load the model
        model = keras.models.load_model(local_path)

        print("✅ Model successfully downloaded and loaded from GCS")
        return model

    except Exception as e:
        print(f"❌ Failed to load model from GCS: {e}")
        return None


from google.cloud import storage
from tensorflow import keras
import os


def save_model_to_gcs(model):
    """
    Save a Keras model locally and upload it to GCS.
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # Ensure local directory exists
    os.makedirs(LOCAL_REGISTRY_PATH, exist_ok=True)
    local_path = os.path.join(LOCAL_REGISTRY_PATH, os.path.basename(BLOB_PATH))

    try:
        # Save model locally
        model.save(local_path)
        print(f"✅ Model saved locally at {local_path}")

        # Upload to GCS
        blob = bucket.blob(BLOB_PATH)
        blob.upload_from_filename(local_path)
        print(f"✅ Model uploaded to GCS at gs://{BUCKET_NAME}/{BLOB_PATH}")

    except Exception as e:
        print(f"❌ Failed to save model to GCS: {e}")
