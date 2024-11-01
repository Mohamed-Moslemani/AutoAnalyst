from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from celery.result import AsyncResult
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import boto3
from botocore.exceptions import ClientError
import tempfile
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

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Initialize S3 client
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, S3_BUCKET_NAME]):
    raise ValueError("One or more AWS environment variables are not set.")

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

def upload_file_to_s3(file: UploadFile, s3_key: str):
    try:
        s3_client.upload_fileobj(file.file, S3_BUCKET_NAME, s3_key)
    except ClientError as e:
        raise

def download_file_from_s3(s3_key: str, local_path: str):
    try:
        s3_client.download_file(S3_BUCKET_NAME, s3_key, local_path)
    except ClientError as e:
        raise

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        s3_key = f"uploads/{file.filename}"
        upload_file_to_s3(file, s3_key)
        task = preprocess_task.delay(s3_key)
        return {"task_id": task.id}
    except Exception as e:
        return {"error": str(e)}

@app.post("/pair_messages/")
async def pair_messages_endpoint(file_key: str = Form(...)):
    try:
        task = pair_messages_task.delay(file_key)
        return {"task_id": task.id}
    except Exception as e:
        return {"error": str(e)}

@app.post("/cs_split/")
async def cs_split_endpoint(file_key: str = Form(...)):
    try:
        task = cs_split_task.delay(file_key)
        return {"task_id": task.id}
    except Exception as e:
        return {"error": str(e)}

@app.post("/sales_split/")
async def sales_split_endpoint(file_key: str = Form(...)):
    try:
        task = sales_split_task.delay(file_key)
        return {"task_id": task.id}
    except Exception as e:
        return {"error": str(e)}

@app.post("/search_messages/")
async def search_messages_endpoint(file_key: str = Form(...), text_column: str = Form(...), searched_text: str = Form(...)):
    try:
        task = search_messages_task.delay(file_key, text_column, searched_text)
        return {"task_id": task.id}
    except Exception as e:
        return {"error": str(e)}

@app.post("/filter_by_chat_id/")
async def filter_by_chat_id_endpoint(file_key: str = Form(...), chat_id: str = Form(...)):
    try:
        task = filter_by_chat_id_task.delay(file_key, chat_id)
        return {"task_id": task.id}
    except Exception as e:
        return {"error": str(e)}

@app.post("/make_readable/")
async def make_readable_endpoint(file_key: str = Form(...)):
    try:
        task = make_readable_task.delay(file_key)
        return {"task_id": task.id}
    except Exception as e:
        return {"error": str(e)}

@app.post("/save_to_csv/")
async def save_to_csv_endpoint(file_key: str = Form(...)):
    try:
        task = save_to_csv_task.delay(file_key)
        return {"task_id": task.id}
    except Exception as e:
        return {"error": str(e)}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    task = AsyncResult(task_id)
    if task.state == 'SUCCESS':
        response = {
            "state": task.state,
            "result": task.result
        }
    elif task.state == 'FAILURE':
        response = {
            "state": task.state,
            "result": {"message": str(task.result)}
        }
    else:
        response = {
            "state": task.state,
            "result": {}
        }
    return JSONResponse(response)

@app.get("/download/{filename}")
async def download_file(filename: str):
    try:
        s3_key = f"processed/{filename}"
        temp_dir = tempfile.gettempdir()
        local_path = os.path.join(temp_dir, filename)
        download_file_from_s3(s3_key, local_path)
        return FileResponse(path=local_path, filename=filename, media_type='application/octet-stream')
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return JSONResponse({"error": "File not found."}, status_code=404)
        else:
            return JSONResponse({"error": "Could not download file."}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": "File not found or could not be downloaded."}, status_code=404)
