# app/tasks.py

import os
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
import pandas as pd
import logging
import boto3
from botocore.exceptions import ClientError
import io
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get Redis URL from environment variable
REDIS_URL = os.getenv('REDIS_URL')
if not REDIS_URL:
    raise ValueError("REDIS_URL environment variable is not set.")

# Initialize Celery
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

# Helper function to generate new S3 keys
def generate_new_key(original_key, suffix):
    base_name = os.path.splitext(os.path.basename(original_key))[0]
    return f"processed/{base_name}_{suffix}"

@celery.task(name='preprocess_task')
def preprocess_task(file_key):
    """
    Task to preprocess the uploaded DataFrame.
    """
    try:
        logger.info(f"Starting preprocess_task with file_key: {file_key}")

        # Download the file from S3
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
        file_contents = response['Body'].read()

        # Read the file into a pandas DataFrame
        if file_key.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_contents))
        elif file_key.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_contents))
        else:
            raise ValueError("Unsupported file format.")

        # Preprocess the DataFrame
        df, message = preprocess_dataframe(df)

        # Serialize the DataFrame
        buffer = io.BytesIO()
        pickle.dump(df, buffer)
        buffer.seek(0)

        # Save the preprocessed DataFrame back to S3
        processed_file_key = generate_new_key(file_key, 'processed.pkl')
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=processed_file_key, Body=buffer.getvalue())

        logger.info(f"Processed file {processed_file_key} uploaded to S3 bucket {S3_BUCKET_NAME}.")

        return {'status': 'success', 'message': message, 'processed_file_key': processed_file_key}
    except ClientError as e:
        logger.error(f"Error accessing S3 in preprocess_task: {e}")
        return {'status': 'failure', 'message': f"Error accessing S3: {e}"}
    except Exception as e:
        logger.error(f"Error in preprocess_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(name='pair_messages_task')
def pair_messages_task(processed_file_key):
    """
    Task to pair messages in the DataFrame.
    """
    try:
        logger.info(f"Starting pair_messages_task with processed_file_key: {processed_file_key}")

        # Download the processed DataFrame from S3
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=processed_file_key)
        df_pickle = response['Body'].read()

        # Load the DataFrame
        df = pickle.loads(df_pickle)

        # Pair messages
        paired_df, message = pair_messages(df)
        if paired_df is None:
            return {'status': 'failure', 'message': message}

        # Serialize the DataFrame
        buffer = io.BytesIO()
        pickle.dump(paired_df, buffer)
        buffer.seek(0)

        # Save the paired DataFrame back to S3
        new_file_key = generate_new_key(processed_file_key, 'paired.pkl')
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=new_file_key, Body=buffer.getvalue())

        logger.info(f"Paired DataFrame saved to {new_file_key} in S3.")

        return {'status': 'success', 'message': message, 'processed_file_key': new_file_key}
    except ClientError as e:
        logger.error(f"Error accessing S3 in pair_messages_task: {e}")
        return {'status': 'failure', 'message': f"Error accessing S3: {e}"}
    except Exception as e:
        logger.error(f"Error in pair_messages_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(name='cs_split_task')
def cs_split_task(processed_file_key):
    """
    Task to split CS chats in the DataFrame.
    """
    try:
        logger.info(f"Starting cs_split_task with processed_file_key: {processed_file_key}")

        # Download the processed DataFrame from S3
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=processed_file_key)
        df_pickle = response['Body'].read()

        # Load the DataFrame
        df = pickle.loads(df_pickle)

        # Split CS chats
        cs_agents_ids = [124760, 396575, 354259, 352740, 178283]
        cs_df, message, success = cs_split(df, cs_agents_ids)
        if not success:
            return {'status': 'failure', 'message': message}

        # Serialize the DataFrame
        buffer = io.BytesIO()
        pickle.dump(cs_df, buffer)
        buffer.seek(0)

        # Save the CS split DataFrame back to S3
        new_file_key = generate_new_key(processed_file_key, 'cs_split.pkl')
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=new_file_key, Body=buffer.getvalue())

        logger.info(f"CS split DataFrame saved to {new_file_key} in S3.")

        return {'status': 'success', 'message': message, 'processed_file_key': new_file_key}
    except ClientError as e:
        logger.error(f"Error accessing S3 in cs_split_task: {e}")
        return {'status': 'failure', 'message': f"Error accessing S3: {e}"}
    except Exception as e:
        logger.error(f"Error in cs_split_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(name='sales_split_task')
def sales_split_task(processed_file_key):
    """
    Task to split Sales chats in the DataFrame.
    """
    try:
        logger.info(f"Starting sales_split_task with processed_file_key: {processed_file_key}")

        # Download the processed DataFrame from S3
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=processed_file_key)
        df_pickle = response['Body'].read()

        # Load the DataFrame
        df = pickle.loads(df_pickle)

        # Split Sales chats
        cs_agents_ids = [124760, 396575, 354259, 352740, 178283]
        sales_df, message, success = sales_split(df, cs_agents_ids)
        if not success:
            return {'status': 'failure', 'message': message}

        # Serialize the DataFrame
        buffer = io.BytesIO()
        pickle.dump(sales_df, buffer)
        buffer.seek(0)

        # Save the Sales split DataFrame back to S3
        new_file_key = generate_new_key(processed_file_key, 'sales_split.pkl')
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=new_file_key, Body=buffer.getvalue())

        logger.info(f"Sales split DataFrame saved to {new_file_key} in S3.")

        return {'status': 'success', 'message': message, 'processed_file_key': new_file_key}
    except ClientError as e:
        logger.error(f"Error accessing S3 in sales_split_task: {e}")
        return {'status': 'failure', 'message': f"Error accessing S3: {e}"}
    except Exception as e:
        logger.error(f"Error in sales_split_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(name='search_messages_task')
def search_messages_task(processed_file_key, text_column, searched_text):
    """
    Task to search messages in the DataFrame.
    """
    try:
        logger.info(f"Starting search_messages_task with processed_file_key: {processed_file_key}, text_column: {text_column}, searched_text: {searched_text}")

        # Download the processed DataFrame from S3
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=processed_file_key)
        df_pickle = response['Body'].read()

        # Load the DataFrame
        df = pickle.loads(df_pickle)

        # Search messages
        search_df, message = search_messages(df, text_column, searched_text)

        # Serialize the DataFrame
        buffer = io.BytesIO()
        pickle.dump(search_df, buffer)
        buffer.seek(0)

        # Save the search result DataFrame back to S3
        new_file_key = generate_new_key(processed_file_key, 'search.pkl')
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=new_file_key, Body=buffer.getvalue())

        logger.info(f"Search results saved to {new_file_key} in S3.")

        return {'status': 'success', 'message': message, 'processed_file_key': new_file_key}
    except ClientError as e:
        logger.error(f"Error accessing S3 in search_messages_task: {e}")
        return {'status': 'failure', 'message': f"Error accessing S3: {e}"}
    except Exception as e:
        logger.error(f"Error in search_messages_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(name='filter_by_chat_id_task')
def filter_by_chat_id_task(processed_file_key, chat_id):
    """
    Task to filter DataFrame by Chat ID.
    """
    try:
        logger.info(f"Starting filter_by_chat_id_task with processed_file_key: {processed_file_key}, chat_id: {chat_id}")

        # Download the processed DataFrame from S3
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=processed_file_key)
        df_pickle = response['Body'].read()

        # Load the DataFrame
        df = pickle.loads(df_pickle)

        # Filter by chat ID
        filtered_df, message, success = filter_by_chat_id(df, chat_id)
        if not success:
            return {'status': 'failure', 'message': message}

        # Serialize the DataFrame
        buffer = io.BytesIO()
        pickle.dump(filtered_df, buffer)
        buffer.seek(0)

        # Save the filtered DataFrame back to S3
        new_file_key = generate_new_key(processed_file_key, 'filtered.pkl')
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=new_file_key, Body=buffer.getvalue())

        logger.info(f"Filtered DataFrame saved to {new_file_key} in S3.")

        return {'status': 'success', 'message': message, 'processed_file_key': new_file_key}
    except ClientError as e:
        logger.error(f"Error accessing S3 in filter_by_chat_id_task: {e}")
        return {'status': 'failure', 'message': f"Error accessing S3: {e}"}
    except Exception as e:
        logger.error(f"Error in filter_by_chat_id_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(name='make_readable_task')
def make_readable_task(processed_file_key):
    """
    Task to make DataFrame readable.
    """
    try:
        logger.info(f"Starting make_readable_task with processed_file_key: {processed_file_key}")

        # Download the processed DataFrame from S3
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=processed_file_key)
        df_pickle = response['Body'].read()

        # Load the DataFrame
        df = pickle.loads(df_pickle)

        # Make readable text
        readable_text, message = make_readable(df)

        # Save the readable text to S3
        readable_file_key = generate_new_key(processed_file_key, 'readable.txt')
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=readable_file_key, Body=readable_text.encode('utf-8'))

        logger.info(f"Readable text saved to {readable_file_key} in S3.")

        return {'status': 'success', 'message': message, 'readable_file_key': readable_file_key}
    except ClientError as e:
        logger.error(f"Error accessing S3 in make_readable_task: {e}")
        return {'status': 'failure', 'message': f"Error accessing S3: {e}"}
    except Exception as e:
        logger.error(f"Error in make_readable_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(name='save_to_csv_task')
def save_to_csv_task(processed_file_key):
    """
    Task to save the current DataFrame to CSV.
    """
    try:
        logger.info(f"Starting save_to_csv_task with processed_file_key: {processed_file_key}")

        # Download the processed DataFrame from S3
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=processed_file_key)
        df_pickle = response['Body'].read()

        # Load the DataFrame
        df = pickle.loads(df_pickle)

        # Convert DataFrame to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Save the CSV to S3
        csv_file_key = generate_new_key(processed_file_key, 'output.csv')
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=csv_file_key, Body=csv_buffer.getvalue())

        logger.info(f"CSV file saved to {csv_file_key} in S3.")

        return {'status': 'success', 'message': 'Data saved to CSV successfully!', 'csv_file_key': csv_file_key}
    except ClientError as e:
        logger.error(f"Error accessing S3 in save_to_csv_task: {e}")
        return {'status': 'failure', 'message': f"Error accessing S3: {e}"}
    except Exception as e:
        logger.error(f"Error in save_to_csv_task: {e}")
        return {'status': 'failure', 'message': str(e)}
