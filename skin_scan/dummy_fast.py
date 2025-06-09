from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd

# ✅ Local helper modules
import preprocessing
import data
from model import load_model_from_gcs

# ✅ Image preprocessing function (resizes to 96x96 for model)
def process_input_image(image: Image.Image) -> np.ndarray:
    """
    Resizes and normalizes a skin lesion PIL image to (96, 96, 3).
    """
    resized_image = image.resize((96, 96))
    image_array = np.array(resized_image).astype("float32") / 255.0
    return image_array

# ✅ FastAPI app setup
app = FastAPI()

# ✅ CORS for frontend (Streamlit) communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any origin for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {"message": "Skin Scan FastAPI is running!"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    body_location: str = Form(...)
):
    try:
        # ✅ Load and preprocess image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_array = process_input_image(image)
        X_image = np.expand_dims(image_array, axis=0)  # Shape: (1, 96, 96, 3)

        # ✅ Create metadata for prediction
        X_pred = pd.DataFrame([{
            "age": age,
            "localization": body_location.lower(),
            "sex": sex.lower()
        }])

        # ✅ Prepare metadata using existing pipeline
        metadata = data.get_metadata_from_bq()
        metadata = preprocessing.preprocess_metadata(metadata)
        X_metadata, _, preprocessor, class_names = preprocessing.prepare_data_for_model(metadata)
        X_metadata_pred_final = preprocessor.transform(X_pred)

        # ✅ Load model and predict
        model = load_model_from_gcs()
        result = model.predict([X_image, X_metadata_pred_final])
        predicted_probs = result[0]  # Shape: (num_classes,)

        # ✅ Format result
        class_prob_dict = dict(zip(class_names, predicted_probs))
        sorted_result = dict(sorted(class_prob_dict.items(), key=lambda x: x[1], reverse=True))
        pretty_result = {k: round(float(v), 3) for k, v in sorted_result.items()}

        return pretty_result

    except Exception as e:
        import traceback
        print("Prediction error:\n", traceback.format_exc())
        return {"error": str(e)}
