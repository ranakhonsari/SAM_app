from typing import Tuple
from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
from sam_backend import sam_backend
from PIL import Image
import io

app = FastAPI()

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    points_x: int = Form(...),
    points_y: int = Form(...)
):
    # Read the uploaded image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Convert the image to numpy format
    image_np = np.array(image)

    # Prepare input point
    input_points = np.array([[points_x, points_y]])

    # Get masks and scores from SAM backend
    masks, scores = sam_backend(image_np, input_points)

    # Return masks and scores
    return {"masks": masks, "scores": scores}
