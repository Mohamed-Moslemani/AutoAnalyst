# app/utils.py

import pandas as pd
import json
import logging
from typing import Tuple, Optional, List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text(content: str) -> str:
    """
    Extracts text from different message types in the content.

    Args:
        content (str): The JSON-formatted string containing message data.

    Returns:
        str: Extracted text based on message type.
    """
    try:
        message_data = json.loads(content)

        message_type = message_data.get('type')
        if message_type == 'whatsapp_template':
            template = message_data.get('template', {})
            components = template.get('components', [])

            extracted_text = [component.get('text') for component in components if 'text' in component]
            if extracted_text:
                return ' '.join(extracted_text)

        elif message_type == 'text':
            return message_data.get('text', '')

        elif message_type == 'attachment':
            attachment_type = message_data.get('attachment', {}).get('type', '')
            if attachment_type == 'audio':
                return 'audio'
            else:
                return attachment_type

        return 'template'

    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to extract text: {e}")
        return 'template'

def preprocess_dataframe(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Preprocesses the DataFrame by cleaning and organizing data.

    Args:
        df (pd.DataFrame): The input DataFrame to preprocess.

    Returns:
        Tuple[Optional[pd.DataFrame], str]: A tuple containing the preprocessed DataFrame (or None if failed)
        and a message indicating the result.
    """
    try:
        if df is None or df.empty:
            raise ValueError("No DataFrame loaded. Please upload a file.")
        
        # Convert 'Date & Time' to datetime and sort
        df['Date & Time'] = pd.to_datetime(df['Date & Time'])
        df = df.sort_values(by=['Contact ID', 'Date & Time']).reset_index(drop=True)
        
        # Remove specific Contact ID
        df = df[df['Contact ID'] != 21794581]
        
        # Extract text from 'Content'
        df['text'] = df['Content'].apply(extract_text)
        
        # Create 'Chat ID' by cumulatively counting changes in 'Contact ID'
        df['Chat ID'] = (df['Contact ID'] != df['Contact ID'].shift()).cumsum()
        
        # Reorder columns: Move 'Chat ID' to front and 'text' after 'Content'
        cols = df.columns.tolist()
        if 'Chat ID' in cols:
            cols.remove('Chat ID')
            cols.insert(0, 'Chat ID')
        if 'text' in cols:
            cols.remove('text')
            content_idx = cols.index('Content') + 1 if 'Content' in cols else len(cols)
            cols.insert(content_idx, 'text')
        df = df[cols]

        # Fill missing values
        df['text'] = df['text'].fillna('template')
        df['Type'] = df['Type'].fillna('normal_text')
        df['Sub Type'] = df['Sub Type'].fillna('normal_text')

        # Drop the first row if it exists (optional based on your data)
        if 0 in df.index:
            df = df.drop(index=0).reset_index(drop=True)

        # Log the columns after preprocessing
        logger.info(f"Columns after preprocessing: {df.columns.tolist()}")

        return df, "DataFrame preprocessed successfully!"

    except Exception as e:
        logger.error(f"Error in preprocess_dataframe: {e}")
        return None, f"Preprocessing failed. Error: {str(e)}"

def pair_messages(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Pairs incoming and outgoing messages in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to pair messages.

    Returns:
        Tuple[Optional[pd.DataFrame], str]: A tuple containing the paired DataFrame (or None if no pairs found)
        and a message indicating the result.
    """
    try:
        if df is None or df.empty:
            raise ValueError("No DataFrame loaded. Please upload and preprocess the file.")

        # Initialize variables
        paired_rows: List[Dict] = []
        current_contact_id: Optional[int] = None
        incoming_messages: List[pd.Series] = []
        outgoing_messages: List[pd.Series] = []
        current_messages: List[pd.Series] = []
        current_direction: Optional[str] = None  # 'incoming' or 'outgoing'

        # Sort the DataFrame by 'Contact ID' and 'Date & Time'
        df = df.sort_values(by=['Contact ID', 'Date & Time']).reset_index(drop=True)

        # Iterate over the rows
        for _, row in df.iterrows():
            contact_id = row['Contact ID']
            message_type = row['Message Type']  # 'incoming' or 'outgoing'

            # If we're on a new contact, reset the buffers
            if contact_id != current_contact_id:
                # Process any collected messages
                if current_messages:
                    if current_direction == 'incoming':
                        incoming_messages.extend(current_messages)
                    else:
                        outgoing_messages.extend(current_messages)
                    # Create a pair if possible
                    if incoming_messages or outgoing_messages:
                        paired_rows.append({
                            'Chat ID': row['Chat ID'],
                            'Contact ID': current_contact_id,
                            'incoming_dates': [msg['Date & Time'] for msg in incoming_messages],
                            'outgoing_dates': [msg['Date & Time'] for msg in outgoing_messages],
                            'incoming_sender_ids': [msg['Sender ID'] for msg in incoming_messages],
                            'outgoing_sender_ids': [msg['Sender ID'] for msg in outgoing_messages],
                            'incoming_texts': [msg['text'] for msg in incoming_messages],
                            'outgoing_texts': [msg['text'] for msg in outgoing_messages],
                        })
                # Reset for new contact
                current_contact_id = contact_id
                incoming_messages = []
                outgoing_messages = []
                current_messages = [row]
                current_direction = message_type
                continue

            # If current_direction is None, set it
            if current_direction is None:
                current_direction = message_type

            if message_type == current_direction:
                current_messages.append(row)
            else:
                # Direction changed
                if current_direction == 'incoming':
                    incoming_messages.extend(current_messages)
                else:
                    outgoing_messages.extend(current_messages)

                # Reset current_messages and set to current message
                current_messages = [row]
                current_direction = message_type

                # If we have both incoming and outgoing messages, create a pair
                if incoming_messages and outgoing_messages:
                    paired_rows.append({
                        'Chat ID': row['Chat ID'],
                        'Contact ID': contact_id,
                        'incoming_dates': [msg['Date & Time'] for msg in incoming_messages],
                        'outgoing_dates': [msg['Date & Time'] for msg in outgoing_messages],
                        'incoming_sender_ids': [msg['Sender ID'] for msg in incoming_messages],
                        'outgoing_sender_ids': [msg['Sender ID'] for msg in outgoing_messages],
                        'incoming_texts': [msg['text'] for msg in incoming_messages],
                        'outgoing_texts': [msg['text'] for msg in outgoing_messages],
                    })
                    incoming_messages = []
                    outgoing_messages = []

        # After iterating, process any remaining messages
        if current_messages:
            if current_direction == 'incoming':
                incoming_messages.extend(current_messages)
            else:
                outgoing_messages.extend(current_messages)

        # If any messages are left, create a pair
        if incoming_messages or outgoing_messages:
            paired_rows.append({
                'Chat ID': row['Chat ID'],
                'Contact ID': current_contact_id,
                'incoming_dates': [msg['Date & Time'] for msg in incoming_messages],
                'outgoing_dates': [msg['Date & Time'] for msg in outgoing_messages],
                'incoming_sender_ids': [msg['Sender ID'] for msg in incoming_messages],
                'outgoing_sender_ids': [msg['Sender ID'] for msg in outgoing_messages],
                'incoming_texts': [msg['text'] for msg in incoming_messages],
                'outgoing_texts': [msg['text'] for msg in outgoing_messages],
            })

        if paired_rows:
            paired_df = pd.DataFrame(paired_rows)
            # Reorder columns
            desired_order = [
                'Chat ID', 'Contact ID', 'incoming_dates', 'outgoing_dates',
                'incoming_sender_ids', 'outgoing_sender_ids',
                'outgoing_texts', 'incoming_texts'
            ]
            # Verify all desired columns exist
            missing_cols = set(desired_order) - set(paired_df.columns)
            if missing_cols:
                logger.warning(f"Missing columns in paired_df: {missing_cols}")
                # Optionally, handle missing columns or fill them with default values
                for col in missing_cols:
                    paired_df[col] = None
            paired_df = paired_df[desired_order]
            return paired_df, "Messages paired successfully!"
        else:
            return None, "No pairs found."

def cs_split(df: pd.DataFrame, cs_agents_ids: List[int]) -> Tuple[Optional[pd.DataFrame], str, bool]:
    """
    Splits CS chats based on agent IDs.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cs_agents_ids (List[int]): List of CS agent IDs.

    Returns:
        Tuple[Optional[pd.DataFrame], str, bool]: A tuple containing the CS DataFrame (or None),
        a message, and a success flag.
    """
    try:
        if 'outgoing_sender_ids' not in df.columns:
            message = "Missing required column: 'outgoing_sender_ids'"
            logger.error(message)
            return None, message, False

        cs_df = df[df['outgoing_sender_ids'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
        ).isin(cs_agents_ids)]
        if cs_df.empty:
            return None, "No CS chats found.", False
        else:
            return cs_df, "CS chats filtered successfully!", True
    except Exception as e:
        logger.error(f"Error in cs_split: {e}")
        return None, str(e), False

def sales_split(df: pd.DataFrame, cs_agents_ids: List[int]) -> Tuple[Optional[pd.DataFrame], str, bool]:
    """
    Splits Sales chats based on agent IDs.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cs_agents_ids (List[int]): List of CS agent IDs.

    Returns:
        Tuple[Optional[pd.DataFrame], str, bool]: A tuple containing the Sales DataFrame (or None),
        a message, and a success flag.
    """
    try:
        if 'outgoing_sender_ids' not in df.columns:
            message = "Missing required column: 'outgoing_sender_ids'"
            logger.error(message)
            return None, message, False

        sales_df = df[~df['outgoing_sender_ids'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
        ).isin(cs_agents_ids)]
        if sales_df.empty:
            return None, "No Sales chats found.", False
        else:
            return sales_df, "Sales chats filtered successfully!", True
    except Exception as e:
        logger.error(f"Error in sales_split: {e}")
        return None, str(e), False

def search_messages(df: pd.DataFrame, text_column: str, searched_text: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Searches for messages containing the specified text.

    Args:
        df (pd.DataFrame): The input DataFrame.
        text_column (str): The column to search within.
        searched_text (str): The text to search for.

    Returns:
        Tuple[Optional[pd.DataFrame], str]: A tuple containing the search result DataFrame (or None)
        and a message indicating the result.
    """
    try:
        if text_column not in df.columns:
            message = f"Column '{text_column}' not found in DataFrame."
            logger.error(message)
            return None, message

        search_df = df[df[text_column].str.contains(searched_text, case=False, na=False)]
        if search_df.empty:
            return None, "No messages found containing the searched text."
        else:
            return search_df, "Messages containing the searched text found successfully!"
    except Exception as e:
        logger.error(f"Error in search_messages: {e}")
        return None, str(e)

def filter_by_chat_id(df: pd.DataFrame, chat_id: int) -> Tuple[Optional[pd.DataFrame], str, bool]:
    """
    Filters the DataFrame by the specified Chat ID.

    Args:
        df (pd.DataFrame): The input DataFrame.
        chat_id (int): The Chat ID to filter by.

    Returns:
        Tuple[Optional[pd.DataFrame], str, bool]: A tuple containing the filtered DataFrame (or None),
        a message, and a success flag.
    """
    try:
        if 'Chat ID' not in df.columns:
            message = "Missing required column: 'Chat ID'"
            logger.error(message)
            return None, message, False

        filtered_df = df[df['Chat ID'] == chat_id]
        if filtered_df.empty:
            return None, "No chats found with the specified Chat ID.", False
        else:
            return filtered_df, "Chats filtered by Chat ID successfully!", True
    except Exception as e:
        logger.error(f"Error in filter_by_chat_id: {e}")
        return None, str(e), False

def make_readable(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Converts the DataFrame into a GPT-readable text format.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Tuple[Optional[str], str]: A tuple containing the readable text (or None)
        and a message indicating the result.
    """
    try:
        if 'incoming_texts' not in df.columns or 'outgoing_texts' not in df.columns:
            message = "Required columns 'incoming_texts' or 'outgoing_texts' are missing."
            logger.error(message)
            return None, message

        readable_text = []
        for _, row in df.iterrows():
            chat_info = f"Chat ID: {row['Chat ID']}\nContact ID: {row['Contact ID']}\n"
            incoming = "Incoming Messages:\n" + '\n'.join([f"- {text}" for text in row['incoming_texts']]) + "\n"
            outgoing = "Outgoing Messages:\n" + '\n'.join([f"- {text}" for text in row['outgoing_texts']]) + "\n"
            readable_text.append(chat_info + incoming + outgoing)
        
        readable_text_str = '\n'.join(readable_text)
        return readable_text_str, "Data made GPT-readable successfully!"
    except Exception as e:
        logger.error(f"Error in make_readable: {e}")
        return None, str(e)
