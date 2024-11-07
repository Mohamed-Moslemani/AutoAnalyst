import pandas as pd
import json
from typing import Tuple, Optional, List, Dict
import ast 
import math 
import numpy as np 

def extract_text(content: str) -> str:
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
            return 'audio' if attachment_type == 'audio' else attachment_type
        return 'template'
    except (json.JSONDecodeError, TypeError):
        return 'template'

def preprocess_dataframe(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        if df is None or df.empty:
            raise ValueError("No DataFrame loaded. Please upload a file.")
        df['Date & Time'] = pd.to_datetime(df['Date & Time'])
        df = df.sort_values(by=['Contact ID', 'Date & Time']).reset_index(drop=True)
        df = df[df['Contact ID'] != 21794581]
        df['text'] = df['Content'].apply(extract_text)
        df['Chat ID'] = (df['Contact ID'] != df['Contact ID'].shift()).cumsum()
        
        cols = df.columns.tolist()
        if 'Chat ID' in cols:
            cols.remove('Chat ID')
            cols.insert(0, 'Chat ID')
        if 'text' in cols:
            cols.remove('text')
            content_idx = cols.index('Content') + 1 if 'Content' in cols else len(cols)
            cols.insert(content_idx, 'text')
        df = df[cols]
        df['text'] = df['text'].fillna('template')
        df['Type'] = df['Type'].fillna('normal_text')
        df['Sub Type'] = df['Sub Type'].fillna('normal_text')
        df['Sender ID'] = df['Sender ID'].fillna(0)
        if 0 in df.index:
            df = df.drop(index=0).reset_index(drop=True)
        return df, "DataFrame preprocessed successfully!"
    except ValueError as ve:
        return None, str(ve)
    except KeyError as ke:
        return None, f"Missing column: {str(ke)}"
    except Exception as e:
        return None, f"Preprocessing failed. Error: {str(e)}"

def pair_messages(df):
    # Check if required columns are present
    required_columns = {'Contact ID', 'Date & Time', 'Message Type', 'text', 'Sender ID'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing_columns)}")

    df = df.sort_values(by=['Contact ID', 'Date & Time']).reset_index(drop=True)
    
    paired_data = []
    current_incoming = []
    current_outgoing = []
    outgoing_sender_ids = []  # To store outgoing sender IDs
    current_contact_id = None

    try:
        for index, row in df.iterrows():
            # Check if Contact ID has changed
            if row['Contact ID'] != current_contact_id:
                if current_incoming or current_outgoing:
                    paired_data.append({
                        'Contact ID': current_contact_id,
                        'incoming_messages': current_incoming,
                        'outgoing_messages': current_outgoing,
                        'outgoing_sender_ids': outgoing_sender_ids
                    })
                # Reset for the new Contact ID
                current_contact_id = row['Contact ID']
                current_incoming = []
                current_outgoing = []
                outgoing_sender_ids = []

            # Accumulate incoming and outgoing messages
            if row['Message Type'] == 'incoming':
                if current_outgoing:  # Finalize pairing if there's a pending outgoing group
                    paired_data.append({
                        'Contact ID': current_contact_id,
                        'incoming_messages': current_incoming,
                        'outgoing_messages': current_outgoing,
                        'outgoing_sender_ids': outgoing_sender_ids
                    })
                    current_incoming = []
                    current_outgoing = []
                    outgoing_sender_ids = []
                current_incoming.append(row['text'])
            elif row['Message Type'] == 'outgoing':
                if current_incoming:  # Collect outgoing messages and Sender ID for current incoming messages
                    current_outgoing.append(row['text'])
                    outgoing_sender_ids.append(int(row['Sender ID']))  # Convert Sender ID to integer
            else:
                raise ValueError(f"Unexpected Message Type at index {index}: {row['Message Type']}")

        # Add the last pair if exists
        if current_incoming or current_outgoing:
            paired_data.append({
                'Contact ID': current_contact_id,
                'incoming_messages': current_incoming,
                'outgoing_messages': current_outgoing,
                'outgoing_sender_ids': outgoing_sender_ids
            })

    except Exception as e:
        print(f"An error occurred during message pairing: {e}")
        raise

    # Convert the paired data to a DataFrame
    paired_df = pd.DataFrame(paired_data)
    return paired_df, "Messages paired successfully!"


def parse_column_list(df: pd.DataFrame, column_name: str) -> pd.Series:
    def safe_eval_and_convert_to_int_list(x):
        try:
            parsed = ast.literal_eval(x) if isinstance(x, str) else x
            if isinstance(parsed, list) and parsed:
                result = []
                for elem in parsed:
                    try:
                        if elem is not None and not (isinstance(elem, float) and np.isnan(elem)):
                            result.append(int(elem))
                        else:
                            result.append(np.nan)
                    except (ValueError, TypeError):
                        result.append(np.nan)
                return result
            return []
        except (ValueError, SyntaxError, TypeError):
            return []

        
    df[column_name] = df[column_name].apply(safe_eval_and_convert_to_int_list)
    print(f"Data types after parsing: {df[column_name].apply(type)}")

    return df

def rows_with_all_elements_in_list(value):
    allowed_ids = [124760, 396575, 354259, 352740, 178283, 398639, 467165, 277476, 464154, 1023356]
    if isinstance(value, list) and value:
        return all(elem in allowed_ids for elem in value)
    return False

def rows_with_all_elements_not_in_list(value):
    allowed_ids = [124760, 396575, 354259, 352740, 178283, 398639, 467165, 277476, 464154, 1023356]
    if isinstance(value, list) and value:
        return all(elem not in allowed_ids for elem in value)
    return False


def cs_split(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str, bool]:
    try:
        if 'outgoing_sender_ids' not in df.columns:
            return None, "Missing required column: 'outgoing_sender_ids'", False
        
        cs_df = parse_column_list(df, 'outgoing_sender_ids')
        cs_df = cs_df[cs_df['outgoing_sender_ids'].apply(rows_with_all_elements_in_list)]
        return (cs_df, "CS chats filtered successfully!", True) if not cs_df.empty else (None, "No CS chats found.", False)
    
    except Exception as e:
        return None, str(e), False
    

def sales_split(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str, bool]:
    try:
        if 'outgoing_sender_ids' not in df.columns:
            return None, "Missing required column: 'outgoing_sender_ids'", False
        
        sales_df = parse_column_list(df, 'outgoing_sender_ids')        
        sales_df = sales_df[df['outgoing_sender_ids'].apply(rows_with_all_elements_not_in_list)]
        return (sales_df, "Sales chats filtered successfully!", True) if not sales_df.empty else (None, "No Sales chats found.", False)
    
    except Exception as e:
        return None, str(e), False

def search_messages(df: pd.DataFrame, text_column: str, searched_text: str) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        if text_column not in df.columns:
            return None, f"Column '{text_column}' not found in DataFrame."
        search_df = df[df[text_column].str.contains(searched_text, case=False, na=False)]
        return (search_df, "Messages containing the searched text found successfully!") if not search_df.empty else (None, "No messages found containing the searched text.")
    except Exception as e:
        return None, str(e)

def filter_by_chat_id(df: pd.DataFrame, chat_id_input: str) -> Tuple[Optional[pd.DataFrame], str, bool]:
    expected_chat_id_column = 'Chat ID'
    if expected_chat_id_column not in df.columns:
        return None, f"The DataFrame does not contain a '{expected_chat_id_column}' column.", False
    df[expected_chat_id_column] = df[expected_chat_id_column].astype(str).str.strip()
    chat_id_input = chat_id_input.strip()
    filtered_df = df[df[expected_chat_id_column] == chat_id_input]
    return (filtered_df, f"Successfully filtered {len(filtered_df)} chats with Chat ID {chat_id_input}.", True) if not filtered_df.empty else (None, "No chats found with the specified Chat ID.", False)

def make_readable(df) -> Tuple[Optional[str], str]:
    if df is None or df.empty:
        return None, "No DataFrame loaded. Please upload and preprocess the file."

    result = ""
    previous_contact_id = None

    # Include the Chat ID at the start
    chat_id = df['Chat ID'].iloc[0] if 'Chat ID' in df.columns else 'N/A'
    result += f"Chat ID: {chat_id}\n\n"

    for _, row in df.iterrows():
        contact_id = row.get('Contact ID', 'N/A')
        incoming_texts = row.get('incoming_texts', '[]')
        outgoing_texts = row.get('outgoing_texts', '[]')

        # Convert string representations of lists to actual lists if necessary
        if isinstance(incoming_texts, str):
            try:
                incoming_texts = ast.literal_eval(incoming_texts)
            except (ValueError, SyntaxError):
                incoming_texts = []  # Fallback if conversion fails
        if isinstance(outgoing_texts, str):
            try:
                outgoing_texts = ast.literal_eval(outgoing_texts)
            except (ValueError, SyntaxError):
                outgoing_texts = []  # Fallback if conversion fails

        # Add a separator when the Contact ID changes
        if contact_id != previous_contact_id and previous_contact_id is not None:
            result += "-" * 70 + "\n"
        if contact_id != previous_contact_id:
            result += f"Contact ID: {contact_id}\n\n"
            previous_contact_id = contact_id

        # Append incoming and outgoing messages in the "Client" and "Agent" format
        for msg in incoming_texts:
            result += f"Client: {msg}\n"
        for msg in outgoing_texts:
            result += f"Agent: {msg}\n"
        result += "\n"

    # Save the result to a file and return the content
    save_file_name = "chat_transcript.txt"
    with open(save_file_name, 'w', encoding='utf-8') as file:
        file.write(result)

    return result, f"Data saved to {save_file_name}"
