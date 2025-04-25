from authenticator import User
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
import requests
from celery import Celery


class Helper:
    def __init__(self):
        load_dotenv()

        # AWS and DB Setup
        AWS_ACCESS_KEY = os.getenv("boto3_aws_access_key_id")
        AWS_SECRET_KEY = os.getenv("boto3_aws_secret_access_key")
        AWS_REGION = os.getenv("boto3_region_name")

        client = MongoClient(os.getenv("db_url"))
        self.db = client["Face_Recognitions"]
        self.video_sources = self.db["video_sources"]
        self.video_status = self.db["video_status"]
        self.current_video_source = None
        self.face_embeddings = self.db["face_embeddings"]

        self.S3_BUCKET_NAME = os.getenv("bucket_name")
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION,
        )
        self.modal = FaceAnalysis(providers=["CPUExecutionProvider"])
        self.modal.prepare(ctx_id=0, det_size=(640, 480))
        self.tolerance = 1.0
        self.firstFrame = None
        self.min_area = 5000 # Minimum area of the contour to be considered as motion
        self.show_frame = os.getenv("show_frame", "false").lower() == "true"
        self.cap = None 
        self.cached_faces = list(self.face_embeddings.find({}))
        self.known_face_embeddings = np.array([np.array(f["embedding"]) for f in self.cached_faces])
    
    def detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.firstFrame is None:
            self.firstFrame = gray
            return False

        delta_frame = cv2.absdiff(self.firstFrame, gray)
        thresh = cv2.threshold(delta_frame, 10, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = imutils.grab_contours(cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))

        for c in cnts:
            if cv2.contourArea(c) >= self.min_area:
                return True
        return False

    def set_video_source(self, source):

        if source == "0":
            self.cap = cv2.VideoCapture(0)  # Use 0 for webcam
            return
        if source.startswith("rtsp://"):
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            return
        # if source.startswith("http://") or source.startswith("https://"):
        #     self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        #     return
        # if source.startswith("s3://"):
        #     s3_path = source[5:]    
        #     bucket_name, object_key = s3_path.split("/", 1)
        #     try:
        #         response = self.s3.get_object(Bucket=bucket_name, Key=object_key)
        #         video_stream = response["Body"].read()
        #         video_array = np.frombuffer(video_stream, np.uint8)
        #         self.cap = cv2.VideoCapture(io.BytesIO(video_array))
        #         return
        #     except (NoCredentialsError, PartialCredentialsError) as e:
        #         print(f"Credentials error: {e}")
        #         return

    def process_video_source(self, source_name=None, user:User=None):
        try:
            if source_name:
                self.current_video_source = source_name
                source_docs = self.video_sources.find_one({"source_name": source_name})
                self.set_video_source(source_docs["rtsp_url"])
            else:
                return "Video source not set."
            
            face_embeddings = list(self.db["face_embeddings"].find({}))
            known_face_embeddings = np.array([np.array(i["embedding"]) for i in face_embeddings])
            delay = 0.01  # Delay in seconds
            print("Processing video source...", source_name)
            while True:

                # 🔄 Control Check (play/pause/stop)
                status_doc = self.get_video_status(source_name, user)
                status = status_doc.get("status", "stopped") if status_doc else "stopped"

                if status == "paused":
                    print(f"Video '{source_name}' is paused.")
                    sleep(1)
                    continue
                elif status == "stopped":
                    print(f"Video '{source_name}' is stopped.")
                    break

                if self.cap is None:
                    print("Video source not set.")
                    break
                success, frame = self.cap.read()
                if not success:
                    print("Failed to capture frame")
                    break

                if not known_face_embeddings.size:
                    continue

                if not self.detect_motion(frame):
                    print("No motion detected")
                    continue

                frame = cv2.resize(frame, (640, 480))
                faces = self.modal.get(frame)

                for face in faces:
                    # start = datetime.now()
                    self.process_face(face, frame, face_embeddings, known_face_embeddings)
                    # end = datetime.now()
                    # print(f"Processing time: {end - start}")

                sleep(delay)

        except Exception as e:
            print(f"Error processing video source: {e}")
        
    def set_video_status(self, source_name, status, current_user: User = None):
        if current_user:
            user_id = current_user.user_id
            self.video_status.update_one(
                {"source_name": source_name, "user_id": user_id},
                {"$set": {"status": status}},
                upsert=True # Create the document if it doesn't exist   
            )
        else:
            # If no user is provided, update the status for all users
            self.video_status.update_one(
                {"source_name": source_name},
                {"$set": {"status": status}},
                upsert=True # Create the document if it doesn't exist   
            )
    def get_video_status(self, source_name, user: User = None):
        if user:
            user_id = user.user_id
            status_doc = self.video_status.find_one({"source_name": source_name, "user_id": user_id})
        else:
            status_doc = self.video_status.find_one({"source_name": source_name})
        if status_doc:
            return status_doc.get("status", "stopped")
        return "stopped"

    def process_face(self, face, frame, face_embeddings = None, known_face_embeddings = None):
        if face_embeddings is None or known_face_embeddings is None:
            face_embeddings = list(self.db["face_embeddings"].find({}))
            known_face_embeddings = np.array([np.array(i["embedding"]) for i in face_embeddings])

        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        normalized_embedding = face.embedding / np.linalg.norm(face.embedding)
        face_distances = np.linalg.norm(known_face_embeddings - normalized_embedding, axis=1)
        best_match_index = np.argmin(face_distances)

        if face_distances[best_match_index] > self.tolerance:
            return

        user = face_embeddings[best_match_index]
        user_id = ObjectId(user["_id"])
        name = user["name"]
        print(user_id, name)

        cv2.putText(frame, name, (box[0] + 6, box[3] - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        today = datetime.today()
        is_today_exist = self.db["Emp_Images"].find_one(
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
                        self.s3.upload_fileobj(bio, self.S3_BUCKET_NAME, bucket_path)

                self.db["Emp_Images"].update_one(
                    {"user_id": user_id},
                    {
                        "$set": {
                            "departure_time": datetime.now(),
                            "departure_picture": f"s3://{self.S3_BUCKET_NAME}/{bucket_path}",
                        }
                    },
                )
                # print(f"Updated departure time for {name}")
                # print(f"triggering callback for {name}")
                self.trigger_callback(name, 0.98, datetime.now().isoformat(),"arrival")
            else:
                bucket_path = f"{name}/{current_date}/arrival.jpg"
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    with io.BytesIO(buffer) as bio:
                        bio.seek(0)
                        self.s3.upload_fileobj(bio, self.S3_BUCKET_NAME, bucket_path)

                self.db["Emp_Images"].insert_one(
                    {
                        "user_id": user_id,
                        "arrival_time": datetime.now(),
                        "arrival_picture": f"s3://{self.S3_BUCKET_NAME}/{bucket_path}",
                    }
                )
                # print(f"triggering callback for {name}")
                self.trigger_callback(name, 0.98, datetime.now().isoformat(),"departure")
                # print(f"Recorded arrival for {name}")
        except Exception as e:
            print(f"Error processing face record: {e}")

    def trigger_callback(self, name: str, confidence: float = 0.98, timestamp: str = None, event: str = "face_recognized"):
        callbacks = self.db["callbacks"]
        callback = callbacks.find_one({"name": self.current_video_source})
        if not callback:
            # print(f"No callback found for {self.current_video_source}")
            return
        
        payload = {
            "event": event,
            "name": name,
            "confidence": confidence,
            "timestamp": timestamp
        }
        
        try:
            # print(f"Triggering callback for {self.current_video_source} {callback['url']} with payload: {payload}")
            response = requests.post(callback["url"], json=payload)
            # callback_logs.insert_one({
            #     "name": name,
            #     "url": callback["url"],
            #     "payload": payload,
            #     "status": response.status_code,
            #     "timestamp": datetime.utcnow()
            # })
            # return {"status": response.status_code, "message": "Callback triggered."}
            # print(response.text)
        except Exception as e:
            # print(e)
            return

    def gen_frames(self, source_name=None): 
        if source_name:
            self.set_video_source(source_name)
        else:
            self.set_video_source(0)
        if self.cap is None:
            return "Video source not set."
        while True:
            success, frame = self.cap.read()
            if not success:
                break

            frame = cv2.resize(frame, (640, 480))

            if not self.known_face_embeddings.size or not self.detect_motion(frame):
                continue

            faces = self.modal.get(frame)
            for face in faces:
                self.process_face(face, frame)

            _, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            
    def celery(self):
        celery = Celery(
            "worker",
            broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
            backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")
        )
        return celery