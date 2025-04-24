import os
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
import numpy as np
import boto3
from modal import get_embeddings
from helpers import Helper
from worker import process_video_task


helper = Helper()
# Load environment variables
load_dotenv()

set_video_status = helper.set_video_source

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

# ===== FastAPI App Setup =====
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

myhelper = Helper()

# ===== Routes =====

@app.get("/")
def homepage(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={"request": request}
    )

@app.post("/register_face")
def register_face(name: str, image: UploadFile = File(...)):
    try:
        image_bytes = image.file.read()
        faces = get_embeddings(image_bytes)
    except Exception as e:
        return PlainTextResponse(status_code=500, content=f"Error processing image: {str(e)}")

    if not faces:
        return PlainTextResponse(status_code=404, content="No face found")

    embedding = faces[0].embedding / np.linalg.norm(faces[0].embedding)

    try:
        inserted_id = face_embeddings.insert_one({
            "name": name,
            "embedding": embedding.tolist()
        }).inserted_id
        return PlainTextResponse(content=f"Face registered successfully: {inserted_id}", status_code=200)
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
    if not source:
        return PlainTextResponse(content="Video source not found.", status_code=404)
    return StreamingResponse(
        myhelper.gen_frames(source["rtsp_url"]),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/process_video_source/{source_name}")
async def process_video_source_by_name(source_name: str):
    task = process_video_task.delay(source_name)
    return {"message": f"Processing for '{source_name}' started.", "task_id": task.id}

@app.post("/video_control/{source_name}")
def control_video(source_name: str, action: str):
    if action not in ["play", "pause", "stop"]:
        return {"error": "Invalid action"}
    set_video_status(source_name, action)
    return {"message": f"{action} command sent to video source '{source_name}'."}
