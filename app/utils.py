# app/utils.py

import pandas as pd
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text(content):
    try:
        message_data = json.loads(content)

        if message_data.get('type') == 'whatsapp_template':
            template = message_data.get('template', {})
            components = template.get('components', [])

            extracted_text = []
            for component in components:
                if 'text' in component:
                    extracted_text.append(component['text'])
            if extracted_text:
                return ' '.join(extracted_text)

        elif message_data.get('type') == 'text':
            return message_data.get('text', '')

        elif message_data.get('type') == 'attachment':
            attachment_type = message_data.get('attachment', {}).get('type', '')
            if attachment_type == 'audio':
                return 'audio'
            else:
                return attachment_type

        return 'template'

    except (json.JSONDecodeError, TypeError) as e:
        return 'template'

def preprocess_dataframe(df):
    try:
        if df is None or df.empty:
            raise ValueError("No DataFrame loaded. Please upload a file.")

        # Parse 'Date & Time' column
        df['Date & Time'] = pd.to_datetime(df['Date & Time'])

        # Sort the DataFrame by 'Contact ID' and 'Date & Time'
        df = df.sort_values(by=['Contact ID', 'Date & Time']).reset_index(drop=True)

        # Remove specific 'Contact ID' if needed
        df = df[df['Contact ID'] != 21794581]

        # Extract text from 'Content' column
        df['text'] = df['Content'].apply(extract_text)

        # Create 'Chat ID' by detecting changes in 'Contact ID'
        df['Chat ID'] = (df['Contact ID'] != df['Contact ID'].shift()).cumsum()

        # Rearrange columns
        cols = df.columns.tolist()
        chat_id_idx = cols.index('Chat ID')
        cols.insert(0, cols.pop(chat_id_idx))  # Move 'Chat ID' to the front
        content_idx = cols.index('Content')
        cols.insert(content_idx + 1, cols.pop(cols.index('text')))  # Move 'text' after 'Content'
        df = df[cols]

        # Fill missing values
        df['text'] = df['text'].fillna('template')
        df['Type'] = df['Type'].fillna('normal_text')
        df['Sub Type'] = df['Sub Type'].fillna('normal_text')

        # Drop the first row if needed
        df = df.drop(index=0).reset_index(drop=True)

        # Log the columns after preprocessing
        logger.info(f"Columns after preprocessing: {df.columns.tolist()}")

        return df, "DataFrame preprocessed successfully!"

    except Exception as e:
        logger.error(f"Error in preprocess_dataframe: {e}")
        return None, f"Preprocessing failed. Error: {str(e)}"

def pair_messages(df):
    try:
        if df is None or df.empty:
            raise ValueError("No DataFrame loaded. Please upload and preprocess the file.")

        # Initialize variables
        paired_rows = []
        current_contact_id = None
        incoming_messages = []
        outgoing_messages = []
        current_messages = []
        current_direction = None  # 'incoming' or 'outgoing'

        # Sort the DataFrame by 'Contact ID' and 'Date & Time'
        df = df.sort_values(by=['Contact ID', 'Date & Time']).reset_index(drop=True)

        # Iterate over the rows
        for index, row in df.iterrows():
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
            paired_df = paired_df[['Chat ID', 'Contact ID', 'incoming_dates', 'outgoing_dates',
       'incoming_sender_ids', 'outgoing_sender_ids', 'outgoing_texts',
       'incoming_texts']]
            return paired_df, "Messages paired successfully!"
        else:
            return None, "No pairs found."

    except Exception as e:
        logger.error(f"Error in pair_messages: {e}")
        return None, f"Pairing failed. Error: {str(e)}"

def cs_split(df, cs_agents_ids):
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

def sales_split(df, cs_agents_ids):
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

def search_messages(df, text_column, searched_text):
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

def filter_by_chat_id(df, chat_id):
    try:
        if 'Chat ID' not in df.columns:
            message = "Missing required column: 'Chat ID'"
            logger.error(message)
            return None, message, False

        filtered_df = df[df['Chat ID'] == int(chat_id)]
        if filtered_df.empty:
            return None, "No chats found with the specified Chat ID.", False
        else:
            return filtered_df, "Chats filtered by Chat ID successfully!", True
    except Exception as e:
        logger.error(f"Error in filter_by_chat_id: {e}")
        return None, str(e), False

def make_readable(df):
    try:
        if 'incoming_texts' not in df.columns or 'outgoing_texts' not in df.columns:
            message = "Required columns 'incoming_texts' or 'outgoing_texts' are missing."
            logger.error(message)
            return None, message

        readable_text = ''
        for index, row in df.iterrows():
            readable_text += f"Chat ID: {row['Chat ID']}\n"
            readable_text += f"Contact ID: {row['Contact ID']}\n"
            readable_text += "Incoming Messages:\n"
            for text in row['incoming_texts']:
                readable_text += f"- {text}\n"
            readable_text += "Outgoing Messages:\n"
            for text in row['outgoing_texts']:
                readable_text += f"- {text}\n"
            readable_text += "\n"
        return readable_text, "Data made GPT-readable successfully!"
    except Exception as e:
        logger.error(f"Error in make_readable: {e}")
        return None, str(e)
