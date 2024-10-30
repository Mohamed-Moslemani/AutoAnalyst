# app/main.py

from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from celery.result import AsyncResult
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):  # Added type annotation
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        upload_folder = "uploads"
        os.makedirs(upload_folder, exist_ok=True)
        file_location = os.path.join(upload_folder, file.filename)
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)
        # Start the preprocess task
        task = preprocess_task.delay(file_location)
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error in /upload/: {e}")
        return {"error": str(e)}

@app.post("/pair_messages/")
async def pair_messages_endpoint(file_path: str = Form(...)):
    try:
        task = pair_messages_task.delay(file_path)
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error in /pair_messages/: {e}")
        return {"error": str(e)}

@app.post("/cs_split/")
async def cs_split_endpoint(file_path: str = Form(...)):
    try:
        task = cs_split_task.delay(file_path)
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error in /cs_split/: {e}")
        return {"error": str(e)}

@app.post("/sales_split/")
async def sales_split_endpoint(file_path: str = Form(...)):
    try:
        task = sales_split_task.delay(file_path)
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error in /sales_split/: {e}")
        return {"error": str(e)}

@app.post("/search_messages/")
async def search_messages_endpoint(file_path: str = Form(...), text_column: str = Form(...), searched_text: str = Form(...)):
    try:
        task = search_messages_task.delay(file_path, text_column, searched_text)
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error in /search_messages/: {e}")
        return {"error": str(e)}

@app.post("/filter_by_chat_id/")
async def filter_by_chat_id_endpoint(file_path: str = Form(...), chat_id: str = Form(...)):
    try:
        task = filter_by_chat_id_task.delay(file_path, chat_id)
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error in /filter_by_chat_id/: {e}")
        return {"error": str(e)}

@app.post("/make_readable/")
async def make_readable_endpoint(file_path: str = Form(...)):
    try:
        task = make_readable_task.delay(file_path)
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error in /make_readable/: {e}")
        return {"error": str(e)}

@app.post("/save_to_csv/")
async def save_to_csv_endpoint(file_path: str = Form(...)):
    try:
        task = save_to_csv_task.delay(file_path)
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
    file_path = os.path.join("processed", filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=filename)
    else:
        return {"error": "File not found"}
