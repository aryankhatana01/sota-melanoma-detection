"""
FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.
To Run the Server:
uvicorn api:app --reload
"""

from typing import Union, List
from fastapi import FastAPI, Query, UploadFile, File
import shutil
from pydantic import BaseModel
from pathlib import Path
from dataset import SIIMISICDataset
import utils
from fastapi.middleware.cors import CORSMiddleware
from cors_origins import Origins
import pandas as pd

app = FastAPI()

origins = Origins.origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_filename = ""

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    path = Path(__file__).parents[1] / "saved_images" / file.filename
    try:
        with path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        print("Error: ", e)
    global current_filename
    current_filename = file.filename
    return {"filename": file.filename}

@app.get("/predict/")
async def predict(filename: str):
    path = Path(__file__).parents[1] / "saved_images" / filename
    # print(type(str(path)))
    path = str(path)
    utils.create_df(path)
    models = utils.get_models_list()
    df_single_image = pd.read_csv('../datasettesting.csv')
    transforms_test = utils.get_test_transforms()
    dataset_test = SIIMISICDataset(df_single_image, 'test', 'test', transform=transforms_test)
    image = dataset_test[0]
    image = image.to('cpu').unsqueeze(0)
    print(image)
    prediction, probs = utils.predict_single_image(image, models)
    # print(len(models))
    # utils.predict_single_image(utils.get_test_transforms()(image=path), models)
    return {"Probs": probs,
            "Prediction": prediction}


@app.get("/")
def read_root():
    return {"Hello": "World"}