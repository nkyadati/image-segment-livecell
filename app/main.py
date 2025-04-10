from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from app.inference import SegFormerInference
from app.utils import image_to_base64
from PIL import Image
import io

app = FastAPI()

# Initialize the SegFormer inference model
model = SegFormerInference()

@app.get("/")
def root():
    """
    Root endpoint to verify that the API is running.

    Returns:
        dict: Status message.
    """
    return {"status": "SegFormer API is running!"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predict the segmentation mask for an uploaded image using a trained SegFormer model.

    Accepts an image file via multipart/form-data, performs semantic segmentation, and returns
    the predicted mask as a PNG image stream.

    Args:
        file (UploadFile): The image file uploaded by the user.

    Returns:
        StreamingResponse: A PNG image of the predicted mask.
        JSONResponse: An error message if prediction fails.
    """
    try:
        # Read and convert the uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Run inference
        mask = model.predict(image)

        # Encode mask to PNG format in memory
        buffer = io.BytesIO()
        mask.save(buffer, format="PNG")
        buffer.seek(0)

        # Return mask as image stream
        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        # Return error response if anything fails
        return JSONResponse(status_code=500, content={"error": str(e)})