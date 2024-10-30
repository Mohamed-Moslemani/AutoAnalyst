# app/tasks.py

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
import os
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery
celery = Celery(
    'tasks',
    broker='redis://localhost:6379/0',  # Update if using a password or different host
    backend='redis://localhost:6379/0'  # Update if using a password or different host
)

@celery.task(bind=True)
def preprocess_task(self, file_path):
    """
    Task to preprocess the uploaded DataFrame.
    """
    try:
        logging.info(f"Starting preprocess_task with file_path: {file_path}")
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format.")

        df, message = preprocess_dataframe(df)
        # Save the preprocessed DataFrame to a pickle file
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        processed_file = os.path.join("processed", f"{base_name}_processed.pkl")
        with open(processed_file, 'wb') as f:
            pickle.dump(df, f)
        return {'status': 'success', 'message': message, 'processed_file': processed_file}
    except Exception as e:
        logging.error(f"Error in preprocess_task: {e}")
        return {'status': 'failure', 'message': str(e)}


@celery.task(bind=True)
def pair_messages_task(self, file_path):
    """
    Task to pair messages in the DataFrame.
    """
    try:
        logging.info(f"Starting pair_messages_task with file_path: {file_path}")
        # Load the DataFrame from the pickle file
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
        paired_df, message = pair_messages(df)
        if paired_df is None:
            return {'status': 'failure', 'message': message}
        # Save the paired DataFrame back to the pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(paired_df, f)
        return {'status': 'success', 'message': message, 'processed_file': file_path}
    except Exception as e:
        logging.error(f"Error in pair_messages_task: {e}")
        return {'status': 'failure', 'message': str(e)}
    
@celery.task(bind=True)
def cs_split_task(self, file_path):
    """
    Task to split CS chats in the DataFrame.
    """
    try:
        logging.info(f"Starting cs_split_task with file_path: {file_path}")
        # Load the DataFrame from the pickle file
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
        cs_agents_ids = [124760, 396575, 354259, 352740, 178283]
        cs_df, message, success = cs_split(df, cs_agents_ids)
        if not success:
            return {'status': 'failure', 'message': message}
        # Save the CS split DataFrame back to the pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(cs_df, f)
        return {'status': 'success', 'message': message, 'processed_file': file_path}
    except Exception as e:
        logging.error(f"Error in cs_split_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True)
def sales_split_task(self, file_path):
    """
    Task to split Sales chats in the DataFrame.
    """
    try:
        logging.info(f"Starting sales_split_task with file_path: {file_path}")
        # Load the DataFrame from the pickle file
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
        cs_agents_ids = [124760, 396575, 354259, 352740, 178283]
        sales_df, message, success = sales_split(df, cs_agents_ids)
        if not success:
            return {'status': 'failure', 'message': message}
        # Save the Sales split DataFrame back to the pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(sales_df, f)
        return {'status': 'success', 'message': message, 'processed_file': file_path}
    except Exception as e:
        logging.error(f"Error in sales_split_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True)
def search_messages_task(self, file_path, text_column, searched_text):
    """
    Task to search messages in the DataFrame.
    """
    try:
        logging.info(f"Starting search_messages_task with file_path: {file_path}, text_column: {text_column}, searched_text: {searched_text}")
        # Load the DataFrame from the pickle file
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
        search_df, message = search_messages(df, text_column, searched_text)
        # Save the search result DataFrame back to the pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(search_df, f)
        return {'status': 'success', 'message': message, 'processed_file': file_path}
    except Exception as e:
        logging.error(f"Error in search_messages_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True)
def filter_by_chat_id_task(self, file_path, chat_id):
    """
    Task to filter DataFrame by Chat ID.
    """
    try:
        logging.info(f"Starting filter_by_chat_id_task with file_path: {file_path}, chat_id: {chat_id}")
        # Load the DataFrame from the pickle file
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
        filtered_df, message, success = filter_by_chat_id(df, chat_id)
        if not success:
            return {'status': 'failure', 'message': message}
        # Save the filtered DataFrame back to the pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(filtered_df, f)
        return {'status': 'success', 'message': message, 'processed_file': file_path}
    except Exception as e:
        logging.error(f"Error in filter_by_chat_id_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True)
def make_readable_task(self, file_path):
    """
    Task to make DataFrame readable.
    """
    try:
        logging.info(f"Starting make_readable_task with file_path: {file_path}")
        # Load the DataFrame from the pickle file
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
        readable_text, message = make_readable(df)
        # Save the readable text to a file
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        readable_file = os.path.join("processed", f"{base_name}_readable.txt")
        with open(readable_file, 'w', encoding='utf-8') as f:
            f.write(readable_text)
        return {'status': 'success', 'message': message, 'readable_file': readable_file}
    except Exception as e:
        logging.error(f"Error in make_readable_task: {e}")
        return {'status': 'failure', 'message': str(e)}

@celery.task(bind=True)
def save_to_csv_task(self, file_path):
    """
    Task to save the current DataFrame to CSV.
    """
    try:
        logging.info(f"Starting save_to_csv_task with file_path: {file_path}")
        # Load the DataFrame from the pickle file
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
        # Save the DataFrame to CSV
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        csv_file = os.path.join("processed", f"{base_name}.csv")
        df.to_csv(csv_file, index=False)
        return {'status': 'success', 'message': 'Data saved to CSV successfully!', 'csv_file': csv_file}
    except Exception as e:
        logging.error(f"Error in save_to_csv_task: {e}")
        return {'status': 'failure', 'message': str(e)}
