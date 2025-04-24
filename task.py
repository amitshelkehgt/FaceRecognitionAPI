from celery_worker import celery_app
from helpers import Helper

myhelper = Helper()

@celery_app.task(bind=True)
def process_video_task(self, source_name):
    try:
        myhelper.process_video_source(source_name)
        return f"Video processing for {source_name} completed"
    except Exception as e:
        return f"Error: {str(e)}"