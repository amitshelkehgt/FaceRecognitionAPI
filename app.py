import os
import io
import cv2
from helpers import Helper
import imutils
import numpy as np
import requests
from datetime import datetime, time
from bson import ObjectId
from pymongo import MongoClient
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import PlainTextResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from modal import get_embeddings, modal
import boto3
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi import BackgroundTasks
from fastapi import Security, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader

load_dotenv()

# ===== MongoDB & AWS Setup =====
client = MongoClient(os.getenv("db_url"))
db = client["Face_Recognitions"]
face_embeddings = db["face_embeddings"]
video_sources = db["video_sources"]
callbacks = db["callbacks"]
callback_logs = db["callback_logs"]

AWS_ACCESS_KEY = os.getenv("boto3_aws_access_key_id")
AWS_SECRET_KEY = os.getenv("boto3_aws_secret_access_key")
AWS_REGION = os.getenv("boto3_region_name")
S3_BUCKET_NAME = os.getenv("bucket_name")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)

# API_KEY = os.getenv("API_KEY")
# API_KEY_NAME = "access_token"

# api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# async def get_api_key(api_key: str = Security(api_key_header)):
#     if api_key == API_KEY:
#         return api_key
#     else:
#         raise HTTPException(status_code=403, detail="Could not validate credentials")

# ===== FastAPI App Setup =====
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
myhelper = Helper()

@app.get("/")
def homepage(request:Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={"request": request}
    )

@app.post("/register_face")
def registerFace(name: str, image: UploadFile = File(...)):
    try:
        image_bytes = image.file.read()
        faces = get_embeddings(image_bytes)
    except Exception as e:
        return PlainTextResponse(status_code=500, content=f"Error processing image: {str(e)}")

    if not faces:
        return PlainTextResponse(status_code=404, content="No face found")

    embedding = faces[0].embedding / np.linalg.norm(faces[0].embedding)

    try:
        id = face_embeddings.insert_one({
            "name": name,
            "embedding": embedding.tolist()
        }).inserted_id
        return PlainTextResponse(content=f"Face registered successfully: {id}", status_code=200)
    except Exception as e:
        return PlainTextResponse(status_code=500, content=f"Error saving to database: {str(e)}")


class VideoSource(BaseModel):
    source_name: str
    rtsp_url: str

@app.post("/register_video_source")
def register_video_source(source: VideoSource):
    video_sources.update_one(
        {"source_name": source.source_name},
        {"$set": {"rtsp_url": source.rtsp_url}},
        upsert=True
    )
    return {"message": f"Video source '{source.source_name}' registered successfully."}

class CallbackRequest(BaseModel):
    name: str
    callback_url: str
    event: str

@app.post("/register_callback")
def register_callback(cb: CallbackRequest):
    callbacks.update_one(
        {"name": cb.name},
        {"$set": {
            "url": cb.callback_url,
            "event": cb.event
        }},
        upsert=True
    )
    return {"message": f"Callback for '{cb.name}' registered with event '{cb.event}'."}


@app.get("/video_feed/{source_name}")
async def video_feed(source_name: str):
    source = video_sources.find_one({"source_name": source_name})
    print(source)
    if not source:
        return PlainTextResponse(content="Video source not found.", status_code=404)
    # todo: parse url
    
    return StreamingResponse(myhelper.gen_frames(source["rtsp_url"]), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/process_video_source/{source_name}")
async def process_video_source_by_name(source_name: str, bgtask:BackgroundTasks):
    
    bgtask.add_task(myhelper.process_video_source, source_name)
    return {"message": f"Processing video source '{source_name}' started."}
    
    