"""FastAPI backend — /predict and /health endpoints."""
import os, sys, io, numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.utils.helpers import load_config
from src.inference.predictor import Predictor

app = FastAPI(title="PlantGuard AI API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

config = load_config()
predictors = {}

def get_predictor(model_type="custom"):
    if model_type not in predictors:
        predictors[model_type] = Predictor(config, model_type)
    return predictors[model_type]

@app.get("/health")
def health():
    return {"status": "healthy", "version": "2.0.0"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = "custom"):
    if model not in ("custom", "efficientnet"):
        raise HTTPException(400, "model must be 'custom' or 'efficientnet'")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    try:
        data = await file.read()
        img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        predictor = get_predictor(model)
        return predictor.predict(img)
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
