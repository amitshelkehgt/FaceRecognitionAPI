# worker.py

import os
from celery import Celery
from helpers import Helper

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

celery = Celery(
    "worker",
    broker=os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@rabbitmq:5672/"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")
)

myhelper = Helper()

@celery.task(name="worker.process_video_task")
def process_video_task(source_name: str):
    print(f"[CELERY TASK] Processing video source: {source_name}")
    myhelper.process_video_source(source_name)
