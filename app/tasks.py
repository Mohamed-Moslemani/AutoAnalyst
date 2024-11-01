import os
import boto3
from botocore.exceptions import ClientError
import pandas as pd
from celery import Celery
from typing import Tuple, Optional
from dotenv import load_dotenv
from app.utils import (
    preprocess_dataframe,
    pair_messages,
    cs_split,
    sales_split,
    search_messages,
    filter_by_chat_id,
    make_readable,
    optimize_dataframe
)
import gc
import uuid

# Load environment variables
load_dotenv()

# Retrieve environment variables
REDIS_URL = os.getenv('REDIS_URL')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# Validate environment variables
if not REDIS_URL:
    raise ValueError("REDIS_URL environment variable is not set.")
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, S3_BUCKET_NAME]):
    raise ValueError("One or more AWS environment variables are not set.")

# Initialize Celery
celery = Celery(
    'tasks',
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_concurrency=2,  # Limit concurrency to 2
)

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

def download_file_from_s3(s3_key: str, local_path: str) -> None:
    try:
        s3_client.download_file(S3_BUCKET_NAME, s3_key, local_path)
    except ClientError as e:
        raise

def upload_file_to_s3(local_path: str, s3_key: str) -> None:
    try:
        s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_key)
    except ClientError as e:
        raise

def remove_local_file(local_path: str) -> None:
    try:
        if os.path.exists(local_path):
            os.remove(local_path)
    except Exception:
        pass

@celery.task(bind=True, time_limit=600, soft_time_limit=550)
def preprocess_task(self, s3_input_key: str) -> dict:
    try:
        local_input_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(s3_input_key)}"
        base_name = os.path.splitext(os.path.basename(s3_input_key))[0]
        processed_s3_key = f"processed/{base_name}_processed.csv"
        local_processed_path = f"/tmp/{base_name}_processed.csv"

        download_file_from_s3(s3_input_key, local_input_path)

        if s3_input_key.endswith('.csv'):
            df = pd.read_csv(local_input_path)
        elif s3_input_key.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(local_input_path)
        else:
            raise ValueError("Unsupported file format. Only CSV and Excel files are supported.")

        df = optimize_dataframe(df)
        df, message = preprocess_dataframe(df)
        if df is None:
            raise ValueError("Preprocessing failed.")
        df = optimize_dataframe(df)
        df.to_csv(local_processed_path, index=False)
        upload_file_to_s3(local_processed_path, processed_s3_key)

        remove_local_file(local_input_path)
        remove_local_file(local_processed_path)
        del df
        gc.collect()

        return {'status': 'success', 'message': message, 'processed_file_s3_key': processed_s3_key}

    except Exception as e:
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True, time_limit=600, soft_time_limit=550)
def pair_messages_task(self, s3_input_key: str) -> dict:
    try:
        local_input_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(s3_input_key)}"
        base_name = os.path.splitext(os.path.basename(s3_input_key))[0]
        paired_s3_key = f"processed/{base_name}_paired.csv"
        local_paired_path = f"/tmp/{base_name}_paired.csv"

        download_file_from_s3(s3_input_key, local_input_path)

        if s3_input_key.endswith('.csv'):
            df = pd.read_csv(local_input_path)
        elif s3_input_key.endswith(('.xlsx', '.xls', '.parquet')):
            df = pd.read_excel(local_input_path)
        else:
            raise ValueError("Unsupported file format for pairing messages. Only CSV and Excel files are supported.")

        df = optimize_dataframe(df)
        paired_df, message = pair_messages(df)
        if paired_df is None:
            raise ValueError(message)

        paired_df['incoming_texts'] = paired_df['incoming_texts'].apply(
            lambda msgs: [msg if isinstance(msg, str) else ''.join(msg) for msg in msgs]
        )
        paired_df['outgoing_texts'] = paired_df['outgoing_texts'].apply(
            lambda msgs: [msg if isinstance(msg, str) else ''.join(msg) for msg in msgs]
        )
        paired_df = optimize_dataframe(paired_df)
        paired_df.to_csv(local_paired_path, index=False)
        upload_file_to_s3(local_paired_path, paired_s3_key)

        remove_local_file(local_input_path)
        remove_local_file(local_paired_path)
        del df, paired_df
        gc.collect()

        return {'status': 'success', 'message': message, 'paired_file_s3_key': paired_s3_key}

    except Exception as e:
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True, time_limit=600, soft_time_limit=550)
def cs_split_task(self, s3_input_key: str) -> dict:
    try:
        local_input_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(s3_input_key)}"
        base_name = os.path.splitext(os.path.basename(s3_input_key))[0]
        cs_split_s3_key = f"processed/{base_name}_cs_split.csv"
        local_cs_split_path = f"/tmp/{base_name}_cs_split.csv"

        download_file_from_s3(s3_input_key, local_input_path)

        if s3_input_key.endswith('.csv'):
            df = pd.read_csv(local_input_path)
        elif s3_input_key.endswith(('.xlsx', '.xls', '.parquet')):
            df = pd.read_excel(local_input_path)
        else:
            raise ValueError("Unsupported file format for CS split. Only CSV and Excel files are supported.")

        df = optimize_dataframe(df)
        cs_agents_ids = [124760, 396575, 354259, 352740, 178283]
        cs_df, message, success = cs_split(df, cs_agents_ids)
        if not success:
            raise ValueError(message)
        cs_df = optimize_dataframe(cs_df)
        cs_df.to_csv(local_cs_split_path, index=False)
        upload_file_to_s3(local_cs_split_path, cs_split_s3_key)

        remove_local_file(local_input_path)
        remove_local_file(local_cs_split_path)
        del df, cs_df
        gc.collect()

        return {'status': 'success', 'message': message, 'cs_split_file_s3_key': cs_split_s3_key}

    except Exception as e:
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True, time_limit=600, soft_time_limit=550)
def sales_split_task(self, s3_input_key: str) -> dict:
    try:
        local_input_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(s3_input_key)}"
        base_name = os.path.splitext(os.path.basename(s3_input_key))[0]
        sales_split_s3_key = f"processed/{base_name}_sales_split.csv"
        local_sales_split_path = f"/tmp/{base_name}_sales_split.csv"

        download_file_from_s3(s3_input_key, local_input_path)

        if s3_input_key.endswith('.csv'):
            df = pd.read_csv(local_input_path)
        elif s3_input_key.endswith(('.xlsx', '.xls', '.parquet')):
            df = pd.read_excel(local_input_path)
        else:
            raise ValueError("Unsupported file format for Sales split. Only CSV and Excel files are supported.")

        df = optimize_dataframe(df)
        cs_agents_ids = [124760, 396575, 354259, 352740, 178283]
        sales_df, message, success = sales_split(df, cs_agents_ids)
        if not success:
            raise ValueError(message)
        sales_df = optimize_dataframe(sales_df)
        sales_df.to_csv(local_sales_split_path, index=False)
        upload_file_to_s3(local_sales_split_path, sales_split_s3_key)

        remove_local_file(local_input_path)
        remove_local_file(local_sales_split_path)
        del df, sales_df
        gc.collect()

        return {'status': 'success', 'message': message, 'sales_split_file_s3_key': sales_split_s3_key}

    except Exception as e:
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True, time_limit=600, soft_time_limit=550)
def search_messages_task(self, s3_input_key: str, text_column: str, searched_text: str) -> dict:
    try:
        local_input_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(s3_input_key)}"
        base_name = os.path.splitext(os.path.basename(s3_input_key))[0]
        search_messages_s3_key = f"processed/{base_name}_search_results.csv"
        local_search_messages_path = f"/tmp/{base_name}_search_results.csv"

        download_file_from_s3(s3_input_key, local_input_path)

        if s3_input_key.endswith('.csv'):
            df = pd.read_csv(local_input_path)
        elif s3_input_key.endswith(('.xlsx', '.xls', '.parquet')):
            df = pd.read_excel(local_input_path)
        else:
            raise ValueError("Unsupported file format for searching messages. Only CSV and Excel files are supported.")

        df = optimize_dataframe(df)
        search_df, message = search_messages(df, text_column, searched_text)
        if search_df is None:
            raise ValueError(message)
        search_df = optimize_dataframe(search_df)
        search_df.to_csv(local_search_messages_path, index=False)
        upload_file_to_s3(local_search_messages_path, search_messages_s3_key)

        remove_local_file(local_input_path)
        remove_local_file(local_search_messages_path)
        del df, search_df
        gc.collect()

        return {'status': 'success', 'message': message, 'search_messages_file_s3_key': search_messages_s3_key}

    except Exception as e:
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True, time_limit=600, soft_time_limit=550)
def filter_by_chat_id_task(self, s3_input_key: str, chat_id: str) -> dict:
    try:
        local_input_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(s3_input_key)}"
        base_name = os.path.splitext(os.path.basename(s3_input_key))[0]
        filter_s3_key = f"processed/{base_name}_filtered_chat_{chat_id}.csv"
        local_filter_path = f"/tmp/{base_name}_filtered_chat_{chat_id}.csv"

        download_file_from_s3(s3_input_key, local_input_path)

        if s3_input_key.endswith('.csv'):
            df = pd.read_csv(local_input_path)
        elif s3_input_key.endswith(('.xlsx', '.xls', '.parquet')):
            df = pd.read_excel(local_input_path)
        else:
            raise ValueError("Unsupported file format for filtering by Chat ID. Only CSV and Excel files are supported.")

        df = optimize_dataframe(df)
        filtered_df, message, success = filter_by_chat_id(df, chat_id)
        if not success:
            raise ValueError(message)
        filtered_df = optimize_dataframe(filtered_df)
        filtered_df.to_csv(local_filter_path, index=False)
        upload_file_to_s3(local_filter_path, filter_s3_key)

        remove_local_file(local_input_path)
        remove_local_file(local_filter_path)
        del df, filtered_df
        gc.collect()

        return {'status': 'success', 'message': message, 'filter_file_s3_key': filter_s3_key}

    except Exception as e:
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True, time_limit=600, soft_time_limit=550)
def make_readable_task(self, s3_input_key: str) -> dict:
    try:
        local_input_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(s3_input_key)}"
        base_name = os.path.splitext(os.path.basename(s3_input_key))[0]
        readable_s3_key = f"processed/{base_name}_readable.txt"
        local_readable_path = f"/tmp/{base_name}_readable.txt"

        download_file_from_s3(s3_input_key, local_input_path)

        if s3_input_key.endswith('.csv'):
            df = pd.read_csv(local_input_path)
        elif s3_input_key.endswith(('.xlsx', '.xls', '.parquet')):
            df = pd.read_excel(local_input_path)
        else:
            raise ValueError("Unsupported file format for making readable. Only CSV and Excel files are supported.")

        df = optimize_dataframe(df)
        readable_text, message = make_readable(df)
        if readable_text is None:
            raise ValueError(message)

        with open(local_readable_path, 'w', encoding='utf-8') as f:
            f.write(readable_text)

        upload_file_to_s3(local_readable_path, readable_s3_key)

        remove_local_file(local_input_path)
        remove_local_file(local_readable_path)
        del df
        gc.collect()

        return {'status': 'success', 'message': message, 'readable_file_s3_key': readable_s3_key}

    except Exception as e:
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True, time_limit=600, soft_time_limit=550)
def save_to_csv_task(self, s3_input_key: str) -> dict:
    try:
        local_input_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(s3_input_key)}"
        base_name = os.path.splitext(os.path.basename(s3_input_key))[0]
        csv_s3_key = f"processed/{base_name}.csv"
        local_csv_path = f"/tmp/{base_name}.csv"

        download_file_from_s3(s3_input_key, local_input_path)

        if s3_input_key.endswith('.csv'):
            df = pd.read_csv(local_input_path)
        elif s3_input_key.endswith(('.xlsx', '.xls', '.parquet')):
            df = pd.read_excel(local_input_path)
        else:
            raise ValueError("Unsupported file format for saving to CSV. Only CSV and Excel files are supported.")

        df = optimize_dataframe(df)
        df.to_csv(local_csv_path, index=False)
        upload_file_to_s3(local_csv_path, csv_s3_key)

        remove_local_file(local_input_path)
        remove_local_file(local_csv_path)
        del df
        gc.collect()

        return {'status': 'success', 'message': 'Data saved to CSV successfully!', 'csv_file_s3_key': csv_s3_key}

    except Exception as e:
        return {'status': 'failure', 'message': str(e)}
