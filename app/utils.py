# app/tasks.py

import os
import logging
import boto3
from botocore.exceptions import ClientError
import pandas as pd
import pickle
import io
from celery import Celery
from .utils import (
    preprocess_dataframe,
    pair_messages,
    cs_split,
    sales_split,
    search_messages,
    filter_by_chat_id,
    make_readable
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery
REDIS_URL = os.getenv('REDIS_URL')
if not REDIS_URL:
    raise ValueError("REDIS_URL environment variable is not set.")

celery = Celery(
    'tasks',
    broker=REDIS_URL,
    backend=REDIS_URL
)

# Celery configuration
celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
)

S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
if not S3_BUCKET_NAME:
    raise ValueError("S3_BUCKET_NAME environment variable is not set.")

def download_file_from_s3(file_key, local_path):
    """
    Downloads a file from S3 to the specified local path.
    """
    try:
        s3_client.download_file(S3_BUCKET_NAME, file_key, local_path)
        logger.info(f"Downloaded {file_key} from S3 to {local_path}.")
    except ClientError as e:
        logger.error(f"Error downloading {file_key} from S3: {e}")
        raise

def upload_file_to_s3(local_path, s3_key):
    """
    Uploads a local file to S3 with the specified S3 key.
    """
    try:
        s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_key)
        logger.info(f"Uploaded {local_path} to S3 as {s3_key}.")
    except ClientError as e:
        logger.error(f"Error uploading {local_path} to S3: {e}")
        raise

def remove_local_file(local_path):
    """
    Removes a local file if it exists.
    """
    try:
        if os.path.exists(local_path):
            os.remove(local_path)
            logger.info(f"Removed local file {local_path}.")
    except Exception as e:
        logger.warning(f"Could not remove local file {local_path}: {e}")

@celery.task(bind=True)
def preprocess_task(self, file_key):
    """
    Task to preprocess the uploaded DataFrame.
    Downloads the file from S3, preprocesses it, and uploads the result back to S3.
    """
    try:
        logger.info(f"Starting preprocess_task with file_key: {file_key}")

        # Define local paths
        local_input_path = f"/tmp/{file_key}"
        base_name = os.path.splitext(os.path.basename(file_key))[0]
        processed_file_key = f"processed/{base_name}_processed.pkl"
        local_processed_path = f"/tmp/{base_name}_processed.pkl"

        # Download the file from S3
        download_file_from_s3(file_key, local_input_path)

        # Read the file into a DataFrame
        if file_key.endswith('.csv'):
            df = pd.read_csv(local_input_path)
        elif file_key.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(local_input_path)
        else:
            raise ValueError("Unsupported file format.")

        # Preprocess the DataFrame
        df, message = preprocess_dataframe(df)
        if df is None:
            raise ValueError("Preprocessing returned None.")

        # Save the preprocessed DataFrame to a pickle file
        with open(local_processed_path, 'wb') as f:
            pickle.dump(df, f)
        logger.info(f"Preprocessed DataFrame saved to {local_processed_path}.")

        # Upload the processed file back to S3
        upload_file_to_s3(local_processed_path, processed_file_key)

        # Clean up local files
        remove_local_file(local_input_path)
        remove_local_file(local_processed_path)

        return {'status': 'success', 'message': message, 'processed_file_key': processed_file_key}
    except Exception as e:
        logger.error(f"Error in preprocess_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True)
def pair_messages_task(self, file_key):
    """
    Task to pair messages in the DataFrame.
    Downloads the preprocessed file from S3, pairs messages, and uploads the result back to S3.
    """
    try:
        logger.info(f"Starting pair_messages_task with file_key: {file_key}")

        # Define local paths
        local_input_path = f"/tmp/{os.path.basename(file_key)}"
        paired_file_key = f"processed/pair_messages_{os.path.basename(file_key)}"
        local_paired_path = f"/tmp/pair_messages_{os.path.basename(file_key)}"

        # Download the preprocessed file from S3
        download_file_from_s3(file_key, local_input_path)

        # Load the DataFrame from the pickle file
        with open(local_input_path, 'rb') as f:
            df = pickle.load(f)

        # Pair messages
        paired_df, message = pair_messages(df)
        if paired_df is None:
            raise ValueError(message)

        # Save the paired DataFrame back to a pickle file
        with open(local_paired_path, 'wb') as f:
            pickle.dump(paired_df, f)
        logger.info(f"Paired DataFrame saved to {local_paired_path}.")

        # Upload the paired file back to S3
        upload_file_to_s3(local_paired_path, paired_file_key)

        # Clean up local files
        remove_local_file(local_input_path)
        remove_local_file(local_paired_path)

        return {'status': 'success', 'message': message, 'paired_file_key': paired_file_key}
    except Exception as e:
        logger.error(f"Error in pair_messages_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True)
def cs_split_task(self, file_key):
    """
    Task to split CS chats in the DataFrame.
    Downloads the paired file from S3, splits CS chats, and uploads the result back to S3.
    """
    try:
        logger.info(f"Starting cs_split_task with file_key: {file_key}")

        # Define local paths
        local_input_path = f"/tmp/{os.path.basename(file_key)}"
        cs_split_file_key = f"processed/cs_split_{os.path.basename(file_key)}"
        local_cs_split_path = f"/tmp/cs_split_{os.path.basename(file_key)}"

        # Download the paired file from S3
        download_file_from_s3(file_key, local_input_path)

        # Load the DataFrame from the pickle file
        with open(local_input_path, 'rb') as f:
            df = pickle.load(f)

        # Define CS agent IDs
        cs_agents_ids = [124760, 396575, 354259, 352740, 178283]

        # Split CS chats
        cs_df, message, success = cs_split(df, cs_agents_ids)
        if not success:
            raise ValueError(message)

        # Save the CS split DataFrame back to a pickle file
        with open(local_cs_split_path, 'wb') as f:
            pickle.dump(cs_df, f)
        logger.info(f"CS split DataFrame saved to {local_cs_split_path}.")

        # Upload the CS split file back to S3
        upload_file_to_s3(local_cs_split_path, cs_split_file_key)

        # Clean up local files
        remove_local_file(local_input_path)
        remove_local_file(local_cs_split_path)

        return {'status': 'success', 'message': message, 'cs_split_file_key': cs_split_file_key}
    except Exception as e:
        logger.error(f"Error in cs_split_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True)
def sales_split_task(self, file_key):
    """
    Task to split Sales chats in the DataFrame.
    Downloads the paired file from S3, splits Sales chats, and uploads the result back to S3.
    """
    try:
        logger.info(f"Starting sales_split_task with file_key: {file_key}")

        # Define local paths
        local_input_path = f"/tmp/{os.path.basename(file_key)}"
        sales_split_file_key = f"processed/sales_split_{os.path.basename(file_key)}"
        local_sales_split_path = f"/tmp/sales_split_{os.path.basename(file_key)}"

        # Download the paired file from S3
        download_file_from_s3(file_key, local_input_path)

        # Load the DataFrame from the pickle file
        with open(local_input_path, 'rb') as f:
            df = pickle.load(f)

        # Define CS agent IDs
        cs_agents_ids = [124760, 396575, 354259, 352740, 178283]

        # Split Sales chats
        sales_df, message, success = sales_split(df, cs_agents_ids)
        if not success:
            raise ValueError(message)

        # Save the Sales split DataFrame back to a pickle file
        with open(local_sales_split_path, 'wb') as f:
            pickle.dump(sales_df, f)
        logger.info(f"Sales split DataFrame saved to {local_sales_split_path}.")

        # Upload the Sales split file back to S3
        upload_file_to_s3(local_sales_split_path, sales_split_file_key)

        # Clean up local files
        remove_local_file(local_input_path)
        remove_local_file(local_sales_split_path)

        return {'status': 'success', 'message': message, 'sales_split_file_key': sales_split_file_key}
    except Exception as e:
        logger.error(f"Error in sales_split_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True)
def search_messages_task(self, file_key, text_column, searched_text):
    """
    Task to search messages in the DataFrame.
    Downloads the paired file from S3, searches messages, and uploads the result back to S3.
    """
    try:
        logger.info(f"Starting search_messages_task with file_key: {file_key}, text_column: {text_column}, searched_text: {searched_text}")

        # Define local paths
        local_input_path = f"/tmp/{os.path.basename(file_key)}"
        search_messages_file_key = f"processed/search_messages_{os.path.basename(file_key)}"
        local_search_messages_path = f"/tmp/search_messages_{os.path.basename(file_key)}"

        # Download the paired file from S3
        download_file_from_s3(file_key, local_input_path)

        # Load the DataFrame from the pickle file
        with open(local_input_path, 'rb') as f:
            df = pickle.load(f)

        # Search messages
        search_df, message = search_messages(df, text_column, searched_text)
        if search_df is None:
            raise ValueError(message)

        # Save the search result DataFrame back to a pickle file
        with open(local_search_messages_path, 'wb') as f:
            pickle.dump(search_df, f)
        logger.info(f"Search messages DataFrame saved to {local_search_messages_path}.")

        # Upload the search messages file back to S3
        upload_file_to_s3(local_search_messages_path, search_messages_file_key)

        # Clean up local files
        remove_local_file(local_input_path)
        remove_local_file(local_search_messages_path)

        return {'status': 'success', 'message': message, 'search_messages_file_key': search_messages_file_key}
    except Exception as e:
        logger.error(f"Error in search_messages_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True)
def filter_by_chat_id_task(self, file_key, chat_id):
    """
    Task to filter DataFrame by Chat ID.
    Downloads the paired file from S3, filters by Chat ID, and uploads the result back to S3.
    """
    try:
        logger.info(f"Starting filter_by_chat_id_task with file_key: {file_key}, chat_id: {chat_id}")

        # Define local paths
        local_input_path = f"/tmp/{os.path.basename(file_key)}"
        filter_file_key = f"processed/filter_by_chat_id_{chat_id}_{os.path.basename(file_key)}"
        local_filter_path = f"/tmp/filter_by_chat_id_{chat_id}_{os.path.basename(file_key)}"

        # Download the paired file from S3
        download_file_from_s3(file_key, local_input_path)

        # Load the DataFrame from the pickle file
        with open(local_input_path, 'rb') as f:
            df = pickle.load(f)

        # Filter by Chat ID
        filtered_df, message, success = filter_by_chat_id(df, chat_id)
        if not success:
            raise ValueError(message)

        # Save the filtered DataFrame back to a pickle file
        with open(local_filter_path, 'wb') as f:
            pickle.dump(filtered_df, f)
        logger.info(f"Filtered DataFrame saved to {local_filter_path}.")

        # Upload the filtered file back to S3
        upload_file_to_s3(local_filter_path, filter_file_key)

        # Clean up local files
        remove_local_file(local_input_path)
        remove_local_file(local_filter_path)

        return {'status': 'success', 'message': message, 'filter_file_key': filter_file_key}
    except Exception as e:
        logger.error(f"Error in filter_by_chat_id_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True)
def make_readable_task(self, file_key):
    """
    Task to make DataFrame readable.
    Downloads the paired file from S3, makes it readable, and uploads the result back to S3.
    """
    try:
        logger.info(f"Starting make_readable_task with file_key: {file_key}")

        # Define local paths
        local_input_path = f"/tmp/{os.path.basename(file_key)}"
        readable_file_key = f"processed/readable_{os.path.splitext(os.path.basename(file_key))[0]}.txt"
        local_readable_path = f"/tmp/readable_{os.path.splitext(os.path.basename(file_key))[0]}.txt"

        # Download the paired file from S3
        download_file_from_s3(file_key, local_input_path)

        # Load the DataFrame from the pickle file
        with open(local_input_path, 'rb') as f:
            df = pickle.load(f)

        # Make DataFrame readable
        readable_text, message = make_readable(df)
        if readable_text is None:
            raise ValueError(message)

        # Save the readable text to a local file
        with open(local_readable_path, 'w', encoding='utf-8') as f:
            f.write(readable_text)
        logger.info(f"Readable text saved to {local_readable_path}.")

        # Upload the readable file back to S3
        upload_file_to_s3(local_readable_path, readable_file_key)

        # Clean up local files
        remove_local_file(local_input_path)
        remove_local_file(local_readable_path)

        return {'status': 'success', 'message': message, 'readable_file_key': readable_file_key}
    except Exception as e:
        logger.error(f"Error in make_readable_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True)
def save_to_csv_task(self, file_key):
    """
    Task to save the current DataFrame to CSV.
    Downloads the paired file from S3, saves it as CSV, and uploads the result back to S3.
    """
    try:
        logger.info(f"Starting save_to_csv_task with file_key: {file_key}")

        # Define local paths
        local_input_path = f"/tmp/{os.path.basename(file_key)}"
        csv_file_key = f"processed/{os.path.splitext(os.path.basename(file_key))[0]}.csv"
        local_csv_path = f"/tmp/{os.path.splitext(os.path.basename(file_key))[0]}.csv"

        # Download the paired file from S3
        download_file_from_s3(file_key, local_input_path)

        # Load the DataFrame from the pickle file
        with open(local_input_path, 'rb') as f:
            df = pickle.load(f)

        # Save the DataFrame to CSV
        df.to_csv(local_csv_path, index=False)
        logger.info(f"DataFrame saved to CSV at {local_csv_path}.")

        # Upload the CSV file back to S3
        upload_file_to_s3(local_csv_path, csv_file_key)

        # Clean up local files
        remove_local_file(local_input_path)
        remove_local_file(local_csv_path)

        return {'status': 'success', 'message': 'Data saved to CSV successfully!', 'csv_file_key': csv_file_key}
    except Exception as e:
        logger.error(f"Error in save_to_csv_task: {e}")
        return {'status': 'failure', 'message': str(e)}
