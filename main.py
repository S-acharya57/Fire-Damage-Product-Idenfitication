from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from pathlib import Path
from model import YOLOModel
import shutil

yolo = YOLOModel()

UPLOAD_FOLDER = Path("./uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

app = FastAPI()

@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    print(f'\n\t\tUPLOADED!!!!')
    try:
        '''
        # Load the uploaded image
        image_data = await image.read()
        image = Image.open(io.BytesIO(image_data))

        # Mock processing logic
        # Replace this with your model inference logic
        items = [
            {
                "name": "Laptop",
                "brand": "Dell",
                "model": "Inspiron 15",
                "price": 699.99,
            },
            {
                "name": "TV",
                "brand": "Samsung",
                "model": "SmartLED",
                "price": 399.99,
            },
        ]

        return JSONResponse(content={"items": items})
        '''
        file_path = UPLOAD_FOLDER / image.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        print(f'Starting to pass into model, {file_path}')
        # Perform YOLO inference
        predictions = yolo.predict(str(file_path))
        print(f'{predictions} \n\t\t\t\tare predictions')
        # Clean up uploaded file
        file_path.unlink()  # Remove file after processing
        return JSONResponse(content={"items": predictions})
    

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)