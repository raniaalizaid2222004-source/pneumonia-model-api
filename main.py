import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf

IMG_SIZE = 128
MODEL_PATH = "pneumonia_model.keras"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

app = FastAPI(
    title="Pneumonia X-Ray Classifier API",
    description="API that classifies chest X-ray images (Normal / Pneumonia) using MobileNetV2",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
def load_model():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.get("/")
def root():
    return {"message": "Pneumonia X-Ray Classifier API is running. Visit /docs to try it out."}


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file (jpg, png, ...)")

    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)

    prediction = model.predict(processed_image)
    probability = float(prediction[0][0])

    predicted_class = CLASS_NAMES[1] if probability >= 0.5 else CLASS_NAMES[0]
    confidence = probability if probability >= 0.5 else 1 - probability

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 2),
        "raw_score": round(probability, 4),
    }
