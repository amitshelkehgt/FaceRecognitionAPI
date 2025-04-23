import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis
from pymongo import MongoClient
from datetime import datetime, time
from bson import ObjectId
from dotenv import load_dotenv
from time import sleep
import imutils
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import io

# Load environment variables
load_dotenv()

# AWS and DB Setup
AWS_ACCESS_KEY = os.getenv("boto3_aws_access_key_id")
AWS_SECRET_KEY = os.getenv("boto3_aws_secret_access_key")
AWS_REGION = os.getenv("boto3_region_name")
S3_BUCKET_NAME = os.getenv("bucket_name")
client = MongoClient(os.getenv("db_url"))
db = client["Face_Recognitions"]

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)

# Capture from camera or RTSP
# cap = cv2.VideoCapture(os.getenv("rtsp_url", cv2.CAP_FFMPEG))
cap = cv2.VideoCapture(os.getenv("rtsp_url"), cv2.CAP_FFMPEG)
# cap = cv2.VideoCapture(0)  # Use 0 for webcam

# Face model setup
modal = FaceAnalysis(providers=["CPUExecutionProvider"])
modal.prepare(ctx_id=0, det_size=(640, 480))

tolerance = 1.0
firstFrame = None
min_area = 5000 # Minimum area of the contour to be considered as motion
show_frame = os.getenv("show_frame", "false").lower() == "true"

def detect_motion(frame):
    global firstFrame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if firstFrame is None:
        firstFrame = gray
        return False

    delta_frame = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(delta_frame, 10, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = imutils.grab_contours(cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
    
    # cnts = imutils.grab_contours(cnts)
    # # print("👌👌👌contours = ", len(cnts))
    # motion = False
    # for c in cnts:
	# 	# if the contour is too small, ignore it
    #     if cv2.contourArea(c) < min_area:
    #         continue
    #     if len(cnts) > 1:
    #         motion = True
    #         # (x, y, w, h) = cv2.boundingRect(c)
    #         # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #         # cv2.putText(frame, "Motion", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    #         break
    # return motion

    for c in cnts:
        if cv2.contourArea(c) >= min_area:
            return True
    return False


def run():
    try:
        face_embeddings = list(db["face_embeddings"].find({}))
        known_face_embeddings = np.array([np.array(i["embedding"]) for i in face_embeddings])
        delay = 0.01  # Delay in seconds

        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame")
                break

            if not known_face_embeddings.size:
                continue

            if not detect_motion(frame):
                print("No motion detected")
                continue

            frame = cv2.resize(frame, (640, 480))
            faces = modal.get(frame)

            for face in faces:
                # start = datetime.now()
                process_face(face, frame, face_embeddings, known_face_embeddings)
                # end = datetime.now()

            if show_frame:
                cv2.imshow("Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            sleep(delay)

    finally:
        cap.release()
        if show_frame:
            cv2.destroyAllWindows()


def process_face(face, frame, face_embeddings, known_face_embeddings):
    box = face.bbox.astype(int)
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    normalized_embedding = face.embedding / np.linalg.norm(face.embedding)
    face_distances = np.linalg.norm(known_face_embeddings - normalized_embedding, axis=1)
    best_match_index = np.argmin(face_distances)

    if face_distances[best_match_index] > tolerance:
        return

    user = face_embeddings[best_match_index]
    user_id = ObjectId(user["_id"])
    name = user["name"]
    print(user_id, name)

    cv2.putText(frame, name, (box[0] + 6, box[3] - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    today = datetime.today()
    is_today_exist = db["Emp_Images"].find_one(
        {
            "user_id": user_id,
            "arrival_time": {
                "$gte": datetime.combine(today, time.min),
                "$lte": datetime.combine(today, time.max),
            },
        }
    )

    current_date = datetime.now().strftime("%Y-%m-%d")

    try:
        if is_today_exist:
            bucket_path = f"{name}/{current_date}/departure.jpg"
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                with io.BytesIO(buffer) as bio:
                    bio.seek(0)
                    s3.upload_fileobj(bio, S3_BUCKET_NAME, bucket_path)

            db["Emp_Images"].update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "departure_time": datetime.now(),
                        "departure_picture": f"s3://{S3_BUCKET_NAME}/{bucket_path}",
                    }
                },
            )
            print(f"Updated departure time for {name}")
        else:
            bucket_path = f"{name}/{current_date}/arrival.jpg"
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                with io.BytesIO(buffer) as bio:
                    bio.seek(0)
                    s3.upload_fileobj(bio, S3_BUCKET_NAME, bucket_path)

            db["Emp_Images"].insert_one(
                {
                    "user_id": user_id,
                    "arrival_time": datetime.now(),
                    "arrival_picture": f"s3://{S3_BUCKET_NAME}/{bucket_path}",
                }
            )
            print(f"Recorded arrival for {name}")
    except Exception as e:
        print(f"Error processing face record: {e}")


# def get_embeddings(image: bytes):
#     np_array = np.frombuffer(image, np.uint8)
#     image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
#     return modal.get(image)
#     # if not faces:
#     #     return None
#     # return faces[0].embedding / np.linalg.norm(faces[0].embedding)

def get_embeddings(image: bytes):
    try:
        np_array = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            print("[ERROR] Failed to decode image")
            return []

        faces = modal.get(image)
        if not faces:
            print("[INFO] No face detected")
            return []

        # Sort by largest bounding box (prominent face)
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

        return faces

    except Exception as e:
        print(f"[ERROR] in get_embeddings: {e}")
        return []



if __name__ == "__main__":
    run()
