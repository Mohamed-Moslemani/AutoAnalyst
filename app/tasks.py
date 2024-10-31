# app/tasks.py

import os
import logging
import boto3
from botocore.exceptions import ClientError
import pandas as pd
from celery import Celery
from typing import Tuple, Optional
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retrieve environment variables
REDIS_URL = os.getenv('REDIS_URL')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# Validate environment variables
missing_vars = []
if not REDIS_URL:
    missing_vars.append("REDIS_URL")
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, S3_BUCKET_NAME]):
    missing_vars.append("One or more AWS environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, S3_BUCKET_NAME")
if missing_vars:
    raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

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
    """
    Downloads a file from S3 to the specified local path.
    """
    try:
        logger.info(f"Downloading {s3_key} from S3 bucket {S3_BUCKET_NAME} to {local_path}")
        s3_client.download_file(S3_BUCKET_NAME, s3_key, local_path)
        logger.info(f"Successfully downloaded {s3_key} to {local_path}")
    except ClientError as e:
        logger.error(f"Failed to download {s3_key} from S3: {e}")
        raise

def upload_file_to_s3(local_path: str, s3_key: str) -> None:
    """
    Uploads a local file to S3 with the specified S3 key.
    """
    try:
        logger.info(f"Uploading {local_path} to S3 bucket {S3_BUCKET_NAME} as {s3_key}")
        s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_key)
        logger.info(f"Successfully uploaded {local_path} to {s3_key}")
    except ClientError as e:
        logger.error(f"Failed to upload {local_path} to S3: {e}")
        raise

def remove_local_file(local_path: str) -> None:
    """
    Removes a local file if it exists.
    """
    try:
        if os.path.exists(local_path):
            os.remove(local_path)
            logger.info(f"Removed local file {local_path}")
    except Exception as e:
        logger.warning(f"Could not remove local file {local_path}: {e}")

@celery.task(bind=True, time_limit=600, soft_time_limit=550)
def preprocess_task(self, file_key: str) -> dict:
    """
    Task to preprocess the uploaded DataFrame.
    Downloads the file from S3, preprocesses it, and uploads the result back to S3.
    """
    try:
        logger.info(f"Starting preprocess_task with S3 key: {file_key}")

        # Define local paths
        local_input_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_key)}"
        base_name = os.path.splitext(os.path.basename(file_key))[0]
        processed_s3_key = f"processed/{base_name}_processed.parquet"  # Use Parquet for efficiency
        local_processed_path = f"/tmp/{base_name}_processed.parquet"

        # Download the file from S3
        download_file_from_s3(file_key, local_input_path)

        # Read the file into a DataFrame
        if file_key.endswith('.csv'):
            df = pd.read_csv(local_input_path)
        elif file_key.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(local_input_path)
        else:
            raise ValueError("Unsupported file format. Only CSV and Excel files are supported.")

        # Optimize DataFrame
        df = optimize_dataframe(df)

        # Preprocess the DataFrame
        df, message = preprocess_dataframe(df)
        if df is None:
            raise ValueError("Preprocessing failed.")

        # Further optimize after preprocessing
        df = optimize_dataframe(df)

        # Save the preprocessed DataFrame to a Parquet file
        df.to_parquet(local_processed_path, index=False)
        logger.info(f"Preprocessed DataFrame saved to {local_processed_path}")

        # Upload the processed file back to S3
        upload_file_to_s3(local_processed_path, processed_s3_key)

        # Clean up local files
        remove_local_file(local_input_path)
        remove_local_file(local_processed_path)

        # Free up memory
        del df
        gc.collect()

        return {'status': 'success', 'message': message, 'processed_file_s3_key': processed_s3_key}

    except Exception as e:
        logger.error(f"Error in preprocess_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True, time_limit=600, soft_time_limit=550)
def pair_messages_task(self, file_key: str) -> dict:
    """
    Task to pair messages in the DataFrame.
    Downloads the file from S3, pairs messages, and uploads the result back to S3.
    """
    try:
        logger.info(f"Starting pair_messages_task with S3 key: {file_key}")

        # Define local paths
        local_input_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_key)}"
        base_name = os.path.splitext(os.path.basename(file_key))[0]
        paired_s3_key = f"processed/{base_name}_paired.parquet"
        local_paired_path = f"/tmp/{base_name}_paired.parquet"

        # Download the file from S3
        download_file_from_s3(file_key, local_input_path)

        # Read the DataFrame from the Parquet file
        df = pd.read_parquet(local_input_path)

        # Optimize DataFrame
        df = optimize_dataframe(df)

        # Pair messages
        paired_df, message = pair_messages(df)
        if paired_df is None:
            raise ValueError(message)

        # Further optimize after pairing
        paired_df = optimize_dataframe(paired_df)

        # Save the paired DataFrame to a Parquet file
        paired_df.to_parquet(local_paired_path, index=False)
        logger.info(f"Paired DataFrame saved to {local_paired_path}")

        # Upload the paired file back to S3
        upload_file_to_s3(local_paired_path, paired_s3_key)

        # Clean up local files
        remove_local_file(local_input_path)
        remove_local_file(local_paired_path)

        # Free up memory
        del df, paired_df
        gc.collect()

        return {'status': 'success', 'message': message, 'paired_file_s3_key': paired_s3_key}

    except Exception as e:
        logger.error(f"Error in pair_messages_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True, time_limit=600, soft_time_limit=550)
def cs_split_task(self, file_key: str) -> dict:
    """
    Task to split CS chats in the DataFrame.
    Downloads the file from S3, splits CS chats, and uploads the result back to S3.
    """
    try:
        logger.info(f"Starting cs_split_task with S3 key: {file_key}")

        # Define local paths
        local_input_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_key)}"
        base_name = os.path.splitext(os.path.basename(file_key))[0]
        cs_split_s3_key = f"processed/{base_name}_cs_split.parquet"
        local_cs_split_path = f"/tmp/{base_name}_cs_split.parquet"

        # Download the file from S3
        download_file_from_s3(file_key, local_input_path)

        # Read the DataFrame from the Parquet file
        df = pd.read_parquet(local_input_path)

        # Optimize DataFrame
        df = optimize_dataframe(df)

        # Define CS agent IDs
        cs_agents_ids = [124760, 396575, 354259, 352740, 178283]

        # Split CS chats
        cs_df, message, success = cs_split(df, cs_agents_ids)
        if not success:
            raise ValueError(message)

        # Further optimize after splitting
        cs_df = optimize_dataframe(cs_df)

        # Save the CS split DataFrame to a Parquet file
        cs_df.to_parquet(local_cs_split_path, index=False)
        logger.info(f"CS split DataFrame saved to {local_cs_split_path}")

        # Upload the CS split file back to S3
        upload_file_to_s3(local_cs_split_path, cs_split_s3_key)

        # Clean up local files
        remove_local_file(local_input_path)
        remove_local_file(local_cs_split_path)

        # Free up memory
        del df, cs_df
        gc.collect()

        return {'status': 'success', 'message': message, 'cs_split_file_s3_key': cs_split_s3_key}

    except Exception as e:
        logger.error(f"Error in cs_split_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True, time_limit=600, soft_time_limit=550)
def sales_split_task(self, file_key: str) -> dict:
    """
    Task to split Sales chats in the DataFrame.
    Downloads the file from S3, splits Sales chats, and uploads the result back to S3.
    """
    try:
        logger.info(f"Starting sales_split_task with S3 key: {file_key}")

        # Define local paths
        local_input_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_key)}"
        base_name = os.path.splitext(os.path.basename(file_key))[0]
        sales_split_s3_key = f"processed/{base_name}_sales_split.parquet"
        local_sales_split_path = f"/tmp/{base_name}_sales_split.parquet"

        # Download the file from S3
        download_file_from_s3(file_key, local_input_path)

        # Read the DataFrame from the Parquet file
        df = pd.read_parquet(local_input_path)

        # Optimize DataFrame
        df = optimize_dataframe(df)

        # Define CS agent IDs
        cs_agents_ids = [124760, 396575, 354259, 352740, 178283]

        # Split Sales chats
        sales_df, message, success = sales_split(df, cs_agents_ids)
        if not success:
            raise ValueError(message)

        # Further optimize after splitting
        sales_df = optimize_dataframe(sales_df)

        # Save the Sales split DataFrame to a Parquet file
        sales_df.to_parquet(local_sales_split_path, index=False)
        logger.info(f"Sales split DataFrame saved to {local_sales_split_path}")

        # Upload the Sales split file back to S3
        upload_file_to_s3(local_sales_split_path, sales_split_s3_key)

        # Clean up local files
        remove_local_file(local_input_path)
        remove_local_file(local_sales_split_path)

        # Free up memory
        del df, sales_df
        gc.collect()

        return {'status': 'success', 'message': message, 'sales_split_file_s3_key': sales_split_s3_key}

    except Exception as e:
        logger.error(f"Error in sales_split_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True, time_limit=600, soft_time_limit=550)
def search_messages_task(self, file_key: str, text_column: str, searched_text: str) -> dict:
    """
    Task to search messages in the DataFrame.
    Downloads the file from S3, searches messages, and uploads the result back to S3.
    """
    try:
        logger.info(f"Starting search_messages_task with S3 key: {file_key}, text_column: {text_column}, searched_text: {searched_text}")

        # Define local paths
        local_input_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_key)}"
        base_name = os.path.splitext(os.path.basename(file_key))[0]
        search_messages_s3_key = f"processed/{base_name}_search_results.parquet"
        local_search_messages_path = f"/tmp/{base_name}_search_results.parquet"

        # Download the file from S3
        download_file_from_s3(file_key, local_input_path)

        # Read the DataFrame from the Parquet file
        df = pd.read_parquet(local_input_path)

        # Optimize DataFrame
        df = optimize_dataframe(df)

        # Search messages
        search_df, message = search_messages(df, text_column, searched_text)
        if search_df is None:
            raise ValueError(message)

        # Further optimize after searching
        search_df = optimize_dataframe(search_df)

        # Save the search result DataFrame to a Parquet file
        search_df.to_parquet(local_search_messages_path, index=False)
        logger.info(f"Search messages DataFrame saved to {local_search_messages_path}")

        # Upload the search messages file back to S3
        upload_file_to_s3(local_search_messages_path, search_messages_s3_key)

        # Clean up local files
        remove_local_file(local_input_path)
        remove_local_file(local_search_messages_path)

        # Free up memory
        del df, search_df
        gc.collect()

        return {'status': 'success', 'message': message, 'search_messages_file_s3_key': search_messages_s3_key}

    except Exception as e:
        logger.error(f"Error in search_messages_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True, time_limit=600, soft_time_limit=550)
def filter_by_chat_id_task(self, file_key: str, chat_id: int) -> dict:
    """
    Task to filter DataFrame by Chat ID.
    Downloads the file from S3, filters by Chat ID, and uploads the result back to S3.
    """
    try:
        logger.info(f"Starting filter_by_chat_id_task with S3 key: {file_key}, chat_id: {chat_id}")

        # Define local paths
        local_input_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_key)}"
        base_name = os.path.splitext(os.path.basename(file_key))[0]
        filter_s3_key = f"processed/{base_name}_filtered_chat_{chat_id}.parquet"
        local_filter_path = f"/tmp/{base_name}_filtered_chat_{chat_id}.parquet"

        # Download the file from S3
        download_file_from_s3(file_key, local_input_path)

        # Read the DataFrame from the Parquet file
        df = pd.read_parquet(local_input_path)

        # Optimize DataFrame
        df = optimize_dataframe(df)

        # Filter by Chat ID
        filtered_df, message, success = filter_by_chat_id(df, chat_id)
        if not success:
            raise ValueError(message)

        # Further optimize after filtering
        filtered_df = optimize_dataframe(filtered_df)

        # Save the filtered DataFrame to a Parquet file
        filtered_df.to_parquet(local_filter_path, index=False)
        logger.info(f"Filtered DataFrame saved to {local_filter_path}")

        # Upload the filtered file back to S3
        upload_file_to_s3(local_filter_path, filter_s3_key)

        # Clean up local files
        remove_local_file(local_input_path)
        remove_local_file(local_filter_path)

        # Free up memory
        del df, filtered_df
        gc.collect()

        return {'status': 'success', 'message': message, 'filter_file_s3_key': filter_s3_key}

    except Exception as e:
        logger.error(f"Error in filter_by_chat_id_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True, time_limit=600, soft_time_limit=550)
def make_readable_task(self, file_key: str) -> dict:
    """
    Task to make DataFrame readable.
    Downloads the file from S3, makes it readable, and uploads the result back to S3.
    """
    try:
        logger.info(f"Starting make_readable_task with S3 key: {file_key}")

        # Define local paths
        local_input_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_key)}"
        base_name = os.path.splitext(os.path.basename(file_key))[0]
        readable_s3_key = f"processed/{base_name}_readable.txt"
        local_readable_path = f"/tmp/{base_name}_readable.txt"

        # Download the file from S3
        download_file_from_s3(file_key, local_input_path)

        # Read the DataFrame from the Parquet file
        df = pd.read_parquet(local_input_path)

        # Optimize DataFrame
        df = optimize_dataframe(df)

        # Make DataFrame readable
        readable_text, message = make_readable(df)
        if readable_text is None:
            raise ValueError(message)

        # Save the readable text to a local file
        with open(local_readable_path, 'w', encoding='utf-8') as f:
            f.write(readable_text)
        logger.info(f"Readable text saved to {local_readable_path}")

        # Upload the readable file back to S3
        upload_file_to_s3(local_readable_path, readable_s3_key)

        # Clean up local files
        remove_local_file(local_input_path)
        remove_local_file(local_readable_path)

        # Free up memory
        del df
        gc.collect()

        return {'status': 'success', 'message': message, 'readable_file_s3_key': readable_s3_key}

    except Exception as e:
        logger.error(f"Error in make_readable_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True, time_limit=600, soft_time_limit=550)
def save_to_csv_task(self, file_key: str) -> dict:
    """
    Task to save the current DataFrame to CSV.
    Downloads the file from S3, saves it as CSV, and uploads the result back to S3.
    """
    try:
        logger.info(f"Starting save_to_csv_task with S3 key: {file_key}")

        # Define local paths
        local_input_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_key)}"
        base_name = os.path.splitext(os.path.basename(file_key))[0]
        csv_s3_key = f"processed/{base_name}.csv"
        local_csv_path = f"/tmp/{base_name}.csv"

        # Download the file from S3
        download_file_from_s3(file_key, local_input_path)

        # Read the DataFrame from the Parquet file
        df = pd.read_parquet(local_input_path)

        # Optimize DataFrame
        df = optimize_dataframe(df)

        # Save the DataFrame to CSV
        df.to_csv(local_csv_path, index=False)
        logger.info(f"DataFrame saved to CSV at {local_csv_path}")

        # Upload the CSV file back to S3
        upload_file_to_s3(local_csv_path, csv_s3_key)

        # Clean up local files
        remove_local_file(local_input_path)
        remove_local_file(local_csv_path)

        # Free up memory
        del df
        gc.collect()

        return {'status': 'success', 'message': 'Data saved to CSV successfully!', 'csv_file_s3_key': csv_s3_key}

    except Exception as e:
        logger.error(f"Error in save_to_csv_task: {e}")
        return {'status': 'failure', 'message': str(e)}
