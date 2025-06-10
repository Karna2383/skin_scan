import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from skin_scan import model as md


from skin_scan import preprocessing
from io import BytesIO
from PIL import Image


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
        image_array = preprocessing.process_input_image(image)
        X_image = np.expand_dims(image_array, axis=0)

        # ✅ Create metadata for prediction
        X_pred = pd.DataFrame([{
            "age": age,
            "localization": body_location.lower(),
            "sex": sex.lower()
        }])

        # ✅ Prepare metadata using existing pipeline
        X_metadata = preprocessing.run_X_pipeline(X_pred)

        # ✅ Load model and predict
        model = md.load_model_from_gcs()
        result = model.predict([X_image, X_metadata])
        predicted_probs = result[0]

        # ✅ Format result
        class_names = {'akiec':'one','bcc':'two','bkl':'three','df':'four','mel':'five','nv':'six','vasc':'seven'}
        class_prob_dict = dict(zip(class_names, predicted_probs))
        sorted_result = dict(sorted(class_prob_dict.items(), key=lambda x: x[1], reverse=True))
        pretty_result = {k: round(float(v), 3) for k, v in sorted_result.items()}

        return pretty_result

    except Exception as e:
        import traceback
        print("Prediction error:\n", traceback.format_exc())
        return {"error": str(e)}
