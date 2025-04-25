from http.client import HTTPException
import os
from typing import Annotated
from authenticator import Token, User, UserInDB, UserRequest, authenticate_user, create_access_token, get_current_active_user, get_password_hash
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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta, timezone
import json
from typing import Annotated
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from pydantic import BaseModel

load_dotenv()
# ===== MongoDB & AWS Setup =====
client = MongoClient(os.getenv("db_url"))
db = client["Face_Recognitions"]
face_embeddings = db["face_embeddings"]
video_sources = db["video_sources"]
callbacks = db["callbacks"]
callback_logs = db["callback_logs"]
users_db = db["users"]
if users_db.count_documents({}) == 0:
    users_db.insert_one({
        "username"          :   "ramesh", 
        "email"             :   "ramesh@demo.com",
        "hashed_password"   :   "$2y$10$XEUomzBHqB9n.XQnRKcLkuEOMoqX.qVRMVu.L3rDeK5tj0dYdb0hu",
        "full_name"         :   "raja ramesh h",
        "disabled"          :   False
    })

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

SECRET_KEY = "ad017bd807664cee26a52b3e041f6962d021d9b1d7316527ce0bcab4154961fc"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES  = 100

# no index route should be added, otherwise it will break api agent /  

@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username,"email":user.email}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")

@app.post("/register_user")
async def register_user(user: UserRequest):
    try:
        user_data = {
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "disabled": user.disabled,
            "hashed_password": get_password_hash(user.password)
        }
        result = db["users"].insert_one(user_data)
        return {"message": f"User {user.username} registered successfully.", "user_id": str(result.inserted_id)}
    except Exception as e:
        return PlainTextResponse(status_code=500, content=f"Error saving user: {str(e)}")
    


@app.post("/register_face")
def register_face(current_user: Annotated[User, Depends(get_current_active_user)],name: str, image: UploadFile = File(...)):
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
def register_video_source(current_user: Annotated[User, Depends(get_current_active_user)], source: VideoSource):
    video_sources.update_one(
        {"source_name": source.source_name, "user_id": current_user.user_id},
        {"$set": {"rtsp_url": source.rtsp_url}},
        upsert=True
    )
    return {"message": f"Video source '{source.source_name}' registered successfully."}

class CallbackRequest(BaseModel):
    name: str
    callback_url: str
    event: str

@app.post("/register_callback")
def register_callback(current_user: Annotated[User, Depends(get_current_active_user)], cb: CallbackRequest):
    callbacks.update_one(
        {"name": cb.name, "user_id": current_user.user_id},
        {"$set": {
            "url": cb.callback_url,
            "event": cb.event
        }},
        upsert=True
    )
    return {"message": f"Callback for '{cb.name}' registered with event '{cb.event}'."}

@app.get("/video_feed/{source_name}")
async def video_feed(current_user: Annotated[User, Depends(get_current_active_user)],source_name: str):
    source = video_sources.find_one({"source_name": source_name,"user_id": current_user.user_id})
    if not source:
        return PlainTextResponse(content="Video source not found.", status_code=404)
    return StreamingResponse(
        myhelper.gen_frames(source["rtsp_url"]),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/process_video_source/{source_name}")
async def process_video_source_by_name(current_user: Annotated[User, Depends(get_current_active_user)],source_name: str):
    task = process_video_task.delay(source_name, current_user)
    return {"message": f"Processing for '{source_name}' started.", "task_id": task.id}


@app.get("/video_control/{source_name}")
def get_control_video_status(current_user: Annotated[User, Depends(get_current_active_user)],source_name: str):
    # if action not in ["play", "pause", "stop"]:
    #     return {"error": "Invalid action"}
    action = myhelper.get_video_source(source_name, current_user)
    return {"status": action, "source":source_name}

@app.post("/video_control/{source_name}")
def control_video(current_user: Annotated[User, Depends(get_current_active_user)],source_name: str, action: str):
    if action not in ["play", "pause", "stop"]:
        return {"error": "Invalid action"}
    myhelper.set_video_source(source_name, action, current_user)
    return {"message": f"{action} command sent to video source '{source_name}'."}
