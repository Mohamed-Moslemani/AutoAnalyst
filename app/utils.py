import pandas as pd
import json
from typing import Tuple, Optional, List, Dict
import ast 

import math
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
        if 0 in df.index:
            df = df.drop(index=0).reset_index(drop=True)
        return df, "DataFrame preprocessed successfully!"
    except ValueError as ve:
        return None, str(ve)
    except KeyError as ke:
        return None, f"Missing column: {str(ke)}"
    except Exception as e:
        return None, f"Preprocessing failed. Error: {str(e)}"

def pair_messages(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        if df is None or df.empty:
            raise ValueError("No DataFrame loaded. Please upload and preprocess the file.")
        
        paired_rows = []
        current_contact_id = None
        incoming_messages = []
        outgoing_messages = []
        current_messages = []
        current_direction = None

        # Data is already sorted, so we can iterate directly
        for _, row in df.iterrows():
            contact_id = row['Contact ID']
            message_type = row['Message Type']
            
            # When the contact ID changes, finalize and save the current conversation
            if contact_id != current_contact_id:
                if current_messages:
                    if current_direction == 'incoming':
                        incoming_messages.extend(current_messages)
                    else:
                        outgoing_messages.extend(current_messages)
                
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
                
                # Reset for the new contact ID
                current_contact_id = contact_id
                incoming_messages = []
                outgoing_messages = []
                current_messages = [row]
                current_direction = message_type
                continue

            # If the message type changes, save the accumulated messages in the correct direction
            if message_type != current_direction:
                if current_direction == 'incoming':
                    incoming_messages.extend(current_messages)
                else:
                    outgoing_messages.extend(current_messages)
                
                current_messages = [row]
                current_direction = message_type
                
                # Pair messages if both incoming and outgoing are available
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
            else:
                # Continue collecting messages of the same type
                current_messages.append(row)

        # Finalize the last contact ID messages
        if current_messages:
            if current_direction == 'incoming':
                incoming_messages.extend(current_messages)
            else:
                outgoing_messages.extend(current_messages)

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

        # Convert paired rows to a DataFrame and ensure column order
        if paired_rows:
            paired_df = pd.DataFrame(paired_rows)
            desired_order = [
                'Chat ID', 'Contact ID', 'incoming_dates', 'outgoing_dates',
                'incoming_sender_ids', 'outgoing_sender_ids',
                'outgoing_texts', 'incoming_texts'
            ]
            missing_cols = set(desired_order) - set(paired_df.columns)
            for col in missing_cols:
                paired_df[col] = None
            paired_df = paired_df[desired_order]
            return paired_df, "Messages paired successfully!"
        else:
            return None, "No pairs found."
    except ValueError as ve:
        return None, str(ve)
    except KeyError as ke:
        return None, f"Missing column: {str(ke)}"
    except Exception as e:
        return None, f"An unexpected error occurred: {str(e)}"
   cs_agents_ids = {124760, 396575, 354259, 352740, 178283, 398639, 467165, 277476, 464154, 1023356}

def parse_sender_ids(sender_ids_str: str) -> List[int]:
    """
    Parses the outgoing_sender_ids string into a flat list of integers.
    - Converts floats to integers.
    - Recursively flattens any nested lists.
    - Excludes non-integer and NaN values.
    """
    def flatten(items):
        for item in items:
            if isinstance(item, list):
                yield from flatten(item)
            elif isinstance(item, (int, float)) and not math.isnan(item):
                yield int(item)
            # Ignore non-numeric and NaN values

    try:
        parsed = ast.literal_eval(sender_ids_str)
        if not isinstance(parsed, list):
            print(f"Warning: Parsed sender_ids is not a list: {parsed}")
            return []
        flat_list = list(flatten(parsed))
        print(f"Debug: Original: {sender_ids_str} => Parsed: {flat_list}")  # Debugging Statement
        return flat_list
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing sender_ids_str '{sender_ids_str}': {e}")  # Debugging Statement
        return []

def cs_split(df: pd.DataFrame, cs_agents_ids: set) -> Tuple[Optional[pd.DataFrame], str, bool]:
    """
    Splits the DataFrame into CS chats by including only chats where outgoing_sender_ids
    contain any CS agent IDs.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        cs_agents_ids (set): Set of CS agent sender IDs.
    
    Returns:
        Tuple[Optional[pd.DataFrame], str, bool]: The CS DataFrame, a message, and a success flag.
    """
    try:
        # Check for the required column
        if 'outgoing_sender_ids' not in df.columns:
            return None, "Missing required column: 'outgoing_sender_ids'", False

        # Parse outgoing_sender_ids to lists of ints
        df['parsed_sender_ids'] = df['outgoing_sender_ids'].apply(parse_sender_ids)

        # Debugging: Check for any lists inside parsed_sender_ids
        problematic_rows = df[df['parsed_sender_ids'].apply(lambda x: any(isinstance(id, list) for id in x))]
        if not problematic_rows.empty:
            print("Error: The following rows have lists within 'parsed_sender_ids':")
            print(problematic_rows[['Chat ID', 'parsed_sender_ids']])
            # Handle or remove these rows as needed
            # For this example, we'll exclude them from cs_df
            df = df[~df.index.isin(problematic_rows.index)]
            print("Excluded rows with lists in 'parsed_sender_ids' from cs_split.")

        # Now, proceed to filter
        cs_df = df[df['parsed_sender_ids'].apply(lambda x: any(id in cs_agents_ids for id in x))]

        if cs_df.empty:
            return None, "No CS chats found.", False

        # Drop the temporary 'parsed_sender_ids' column
        cs_df = cs_df.drop(columns=['parsed_sender_ids'])

        return cs_df, "CS chats filtered successfully!", True

    except Exception as e:
        return None, f"Error in cs_split: {str(e)}", False

def sales_split(df: pd.DataFrame, cs_agents_ids: set) -> Tuple[Optional[pd.DataFrame], str, bool]:
    """
    Splits the DataFrame into sales chats by excluding chats where outgoing_sender_ids
    contain any CS agent IDs.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        cs_agents_ids (set): Set of CS agent sender IDs.
    
    Returns:
        Tuple[Optional[pd.DataFrame], str, bool]: The sales DataFrame, a message, and a success flag.
    """
    try:
        if 'parsed_sender_ids' not in df.columns:
            return None, "Missing required column: 'parsed_sender_ids'", False

        # Track rows with missing sender IDs
        missing_sender_id_count = df['parsed_sender_ids'].apply(lambda x: len(x) == 0).sum()

        # Debugging: Check for any lists inside parsed_sender_ids
        problematic_rows = df[df['parsed_sender_ids'].apply(lambda x: any(isinstance(id, list) for id in x))]
        if not problematic_rows.empty:
            print("Error: The following rows have lists within 'parsed_sender_ids':")
            print(problematic_rows[['Chat ID', 'parsed_sender_ids']])
            # Handle or remove these rows as needed
            # For this example, we'll exclude them from sales_df
            df = df[~df.index.isin(problematic_rows.index)]
            print("Excluded rows with lists in 'parsed_sender_ids' from sales_split.")

        # Filter out rows where any sender ID is in cs_agents_ids
        sales_df = df[~df['parsed_sender_ids'].apply(lambda x: any(id in cs_agents_ids for id in x))]

        # Exclude rows with empty parsed_sender_ids from sales_df directly
        sales_df = sales_df[sales_df['parsed_sender_ids'].apply(lambda x: len(x) > 0)]

        if sales_df.empty:
            message = (
                f"No Sales chats found. Additionally, {missing_sender_id_count} rows have no sender id."
                if missing_sender_id_count > 0
                else "No Sales chats found."
            )
            return None, message, False

        message = (
            f"Sales chats filtered successfully! {missing_sender_id_count} rows have no sender id found."
            if missing_sender_id_count > 0
            else "Sales chats filtered successfully!"
        )

        return sales_df, message, True

    except Exception as e:
        return None, f"Error in sales_split: {str(e)}", False

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

def make_readable(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    try:
        output = ""
        previous_contact_id = None

        for index, row in df.iterrows():
            chat_id = row.get('Chat ID', 'N/A')
            contact_id = row.get('Contact ID', 'N/A')
            outgoing_sender_ids = row.get('outgoing_sender_ids', '[]')
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
            if isinstance(outgoing_sender_ids, str):
                try:
                    outgoing_sender_ids = ast.literal_eval(outgoing_sender_ids)
                except (ValueError, SyntaxError):
                    outgoing_sender_ids = []  # Fallback if conversion fails

            # Extract first sender IDs as integers (handling floats and NaNs)
            parsed_outgoing_sender_ids = []
            for sender_id in outgoing_sender_ids:
                if isinstance(sender_id, float) and not math.isnan(sender_id):
                    parsed_outgoing_sender_ids.append(int(sender_id))
                elif isinstance(sender_id, int):
                    parsed_outgoing_sender_ids.append(sender_id)
                else:
                    # Handle non-integer and NaN sender IDs
                    parsed_outgoing_sender_ids.append("No sender ID found")

            # Only show Chat ID, Contact ID if the Contact ID changes
            if contact_id != previous_contact_id:
                if previous_contact_id is not None:
                    output += "-" * 70 + "\n"  # Separator only if the contact ID changes
                output += f"Chat ID: {chat_id}\nContact ID: {contact_id}\n\n"
                previous_contact_id = contact_id

            # Determine the number of messages to iterate through
            max_messages = max(len(incoming_texts), len(outgoing_texts))

            for i in range(max_messages):
                # Add Incoming Message
                if i < len(incoming_texts):
                    msg = incoming_texts[i]
                    if isinstance(msg, str):
                        msg = msg.strip()
                        output += f"Incoming Message: '{msg}'\n"
                    else:
                        output += "Incoming Message: 'Invalid message format'\n"
                else:
                    output += "Incoming Message: 'No message'\n"

                # Add Outgoing Message with Sender ID
                if i < len(outgoing_texts):
                    msg = outgoing_texts[i]
                    if i < len(parsed_outgoing_sender_ids):
                        sender_id = parsed_outgoing_sender_ids[i]
                    else:
                        sender_id = "No sender ID found"

                    if isinstance(msg, str):
                        msg = msg.strip()
                        output += f'Outgoing Message - Sender ID: {sender_id}: "{msg}"\n\n'
                    else:
                        output += f'Outgoing Message - Sender ID: {sender_id}: "Invalid message format"\n\n'
                else:
                    output += 'Outgoing Message - Sender ID: No sender ID found: "No message"\n\n'

        return output, "Data made readable successfully!"
    except Exception as e:
        return None, f"Error making data readable: {str(e)}"
    

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    try:
        for col in df.select_dtypes(include=['int', 'float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='unsigned')
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')
                if col in ['text', 'Type', 'Sub Type']:
                    new_categories = ['normal_text', 'template']
                    df[col] = df[col].cat.add_categories([cat for cat in new_categories if cat not in df[col].cat.categories])
        return df
    except Exception:
        return df
