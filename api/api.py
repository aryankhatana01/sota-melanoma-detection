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
import utils
from fastapi.middleware.cors import CORSMiddleware
from cors_origins import Origins

app = FastAPI()

origins = Origins.origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    path = Path(__file__).parents[1] / "saved_images" / file.filename
    try:
        with path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        print("Error: ", e)

    return {"filename": file.filename}


@app.get("/")
def read_root():
    return {"Hello": "World"}