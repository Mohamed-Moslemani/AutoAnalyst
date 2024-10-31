# app/main.py

from fastapi import FastAPI, Request, File, UploadFile, Form, Response
from fastapi.responses import HTMLResponse, JSONResponse
from celery.result import AsyncResult
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import logging
import boto3
from botocore.exceptions import ClientError
from .tasks import (
    preprocess_task,
    pair_messages_task,
    cs_split_task,
    sales_split_task,
    search_messages_task,
    filter_by_chat_id_task,
    make_readable_task,
    save_to_csv_task
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
)

S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Read file contents
        file_contents = await file.read()

        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=file.filename,
            Body=file_contents
        )

        logger.info(f"File {file.filename} uploaded to S3 bucket {S3_BUCKET_NAME}.")

        # Start the preprocess task
        task = preprocess_task.delay(file.filename)
        return {"task_id": task.id}
    except ClientError as e:
        logger.error(f"Error uploading file to S3: {e}")
        return {"error": f"Error uploading file to S3: {e}"}
    except Exception as e:
        logger.error(f"Error in /upload/: {e}")
        return {"error": str(e)}

@app.post("/pair_messages/")
async def pair_messages_endpoint(file_key: str = Form(...)):
    try:
        task = pair_messages_task.delay(file_key)
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error in /pair_messages/: {e}")
        return {"error": str(e)}

@app.post("/cs_split/")
async def cs_split_endpoint(file_key: str = Form(...)):
    try:
        task = cs_split_task.delay(file_key)
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error in /cs_split/: {e}")
        return {"error": str(e)}

@app.post("/sales_split/")
async def sales_split_endpoint(file_key: str = Form(...)):
    try:
        task = sales_split_task.delay(file_key)
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error in /sales_split/: {e}")
        return {"error": str(e)}

@app.post("/search_messages/")
async def search_messages_endpoint(
    file_key: str = Form(...),
    text_column: str = Form(...),
    searched_text: str = Form(...)
):
    try:
        task = search_messages_task.delay(file_key, text_column, searched_text)
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error in /search_messages/: {e}")
        return {"error": str(e)}

@app.post("/filter_by_chat_id/")
async def filter_by_chat_id_endpoint(
    file_key: str = Form(...),
    chat_id: str = Form(...)
):
    try:
        task = filter_by_chat_id_task.delay(file_key, chat_id)
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error in /filter_by_chat_id/: {e}")
        return {"error": str(e)}

@app.post("/make_readable/")
async def make_readable_endpoint(file_key: str = Form(...)):
    try:
        task = make_readable_task.delay(file_key)
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error in /make_readable/: {e}")
        return {"error": str(e)}

@app.post("/save_to_csv/")
async def save_to_csv_endpoint(file_key: str = Form(...)):
    try:
        task = save_to_csv_task.delay(file_key)
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error in /save_to_csv/: {e}")
        return {"error": str(e)}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    task = AsyncResult(task_id)
    if task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'status': str(task.info.get('status', '')),
            'message': task.info.get('message', ''),
            'result': task.info
        }
    elif task.state == 'FAILURE':
        response = {
            'state': task.state,
            'status': str(task.info),
            'message': task.info.get('message', str(task.info))
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info)
        }
    return JSONResponse(response)

@app.get("/download/{filename}")
async def download_file(filename: str):
    try:
        # Download the processed file from S3
        response = s3_client.get_object(
            Bucket=S3_BUCKET_NAME,
            Key=f"processed/{filename}"
        )
        file_contents = response['Body'].read()

        # Return the file as a streaming response
        return Response(
            content=file_contents,
            media_type='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"'
            }
        )
    except ClientError as e:
        logger.error(f"Error downloading file from S3: {e}")
        return {"error": f"Error downloading file from S3: {e}"}
    except Exception as e:
        logger.error(f"Error in /download/{filename}: {e}")
        return {"error": str(e)}
