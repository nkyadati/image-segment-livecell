import base64
from io import BytesIO
from PIL import Image

def image_to_base64(img: Image.Image) -> str:
    """Convert PIL image to base64-encoded PNG string."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")