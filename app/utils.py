import ast
import json
from typing import Tuple, Optional, List, Dict
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1) TEXT EXTRACTION
# ----------------------------------------------------------------------
def extract_text(content: str) -> str:
    """
    Normalise the JSON blob in the “Content” column to plain text.

    * whatsapp_template  -> concat all component.text fields
    * text               -> text field
    * attachment         -> 'audio' / '<type>'
    * anything else      -> 'template'
    """
    try:
        msg = json.loads(content)
        mtype = msg.get("type", "").lower()

        if mtype == "whatsapp_template":
            comps = msg.get("template", {}).get("components", [])
            return " ".join(c.get("text", "") for c in comps if "text" in c) or "template"

        if mtype == "text":
            return msg.get("text", "")

        if mtype == "attachment":
            atype = msg.get("attachment", {}).get("type", "")
            return "audio" if atype == "audio" else atype

        return "template"

    except (json.JSONDecodeError, TypeError):
        return "template"


# ----------------------------------------------------------------------
# 2) PRE‑PROCESS
# ----------------------------------------------------------------------
def preprocess_dataframe(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
    """
    • parse date, sort inside contact
    • drop internal contact 21794581
    • add 'text' column and incremental Chat ID
    • fill NA values that break later code
    """
    try:
        if df is None or df.empty:
            raise ValueError("No DataFrame loaded.")

        df["Date & Time"] = pd.to_datetime(df["Date & Time"], errors="coerce")
        df = df.sort_values(["Contact ID", "Date & Time"]).reset_index(drop=True)

        df = df[df["Contact ID"] != 21794581]

        df["text"] = df["Content"].apply(extract_text)
        df["Chat ID"] = (df["Contact ID"] != df["Contact ID"].shift()).cumsum()

        # nice column order: Chat ID first, text right after Content
        cols = df.columns.tolist()
        cols.insert(0, cols.pop(cols.index("Chat ID")))
        cols.insert(cols.index("Content") + 1, cols.pop(cols.index("text")))
        df = df[cols]

        # NA housekeeping
        df["text"].fillna("template", inplace=True)
        df["Type"].fillna("normal_text", inplace=True)
        df["Sub Type"].fillna("normal_text", inplace=True)
        df["Sender ID"].fillna(0, inplace=True)

        return df, "DataFrame preprocessed successfully!"

    except Exception as e:
        return None, f"Preprocessing failed – {e}"


# ----------------------------------------------------------------------
# 3) PAIR MESSAGES  (for later GPT‑readable transcript)
# ----------------------------------------------------------------------
def pair_messages(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Build one row per conversation turn (all consecutive incoming msgs
    followed by all consecutive outgoing msgs) *per Chat ID*.
    Lists are kept in chronological order.
    """
    try:
        if df is None or df.empty:
            raise ValueError("No DataFrame loaded.")

        required = {"Chat ID", "Message Type", "Date & Time", "Sender ID", "text"}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns: {required - set(df.columns)}")

        paired_rows: List[Dict] = []

        for chat_id, grp in df.groupby("Chat ID"):
            grp = grp.sort_values("Date & Time")
            buffer: List[Dict] = []
            buffer_dir: Optional[str] = None  # 'incoming' / 'outgoing'

            def flush():
                nonlocal buffer, buffer_dir
                if not buffer:
                    return
                paired_rows.append(
                    {
                        "Chat ID": chat_id,
                        "incoming_dates": [r["Date & Time"] for r in buffer if r["Message Type"] == "incoming"],
                        "outgoing_dates": [r["Date & Time"] for r in buffer if r["Message Type"] == "outgoing"],
                        "incoming_sender_ids": [r["Sender ID"] for r in buffer if r["Message Type"] == "incoming"],
                        "outgoing_sender_ids": [r["Sender ID"] for r in buffer if r["Message Type"] == "outgoing"],
                        "incoming_texts": [r["text"] for r in buffer if r["Message Type"] == "incoming"],
                        "outgoing_texts": [r["text"] for r in buffer if r["Message Type"] == "outgoing"],
                    }
                )
                buffer, buffer_dir = [], None

            for _, row in grp.iterrows():
                direction = row["Message Type"]
                if buffer_dir is None or direction == buffer_dir:
                    buffer.append(row)
                    buffer_dir = direction
                else:
                    flush()
                    buffer.append(row)
                    buffer_dir = direction

            flush()  # last turn in chat

        if not paired_rows:
            return None, "No pairs found."

        paired_df = pd.DataFrame(paired_rows)
        return paired_df, "Messages paired successfully!"

    except Exception as e:
        return None, f"Pairing failed – {e}"


# ----------------------------------------------------------------------
# 4) CS / SALES SPLIT  (unchanged except for safer parsing)
# ----------------------------------------------------------------------
def _safe_eval_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        out = ast.literal_eval(x)
        return out if isinstance(out, list) else []
    except Exception:
        return []


def parse_column_list(df: pd.DataFrame, column_name: str) -> None:
    df[column_name] = df[column_name].apply(_safe_eval_list).astype(object)


_ALLOWED_AGENT_IDS = {
    124760, 396575, 354259, 352740, 178283,
    398639, 467165, 277476, 464154, 1023356,
}


def rows_with_all_elements_not_in_list(value):
    return isinstance(value, list) and all(v not in _ALLOWED_AGENT_IDS for v in value)


def rows_with_all_elements_in_list(value):
    return isinstance(value, list) and all(v in _ALLOWED_AGENT_IDS for v in value)


def cs_split(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str, bool]:
    if "outgoing_sender_ids" not in df.columns:
        return None, "Missing column 'outgoing_sender_ids'", False

    parse_column_list(df, "outgoing_sender_ids")
    cs_df = df[df["outgoing_sender_ids"].apply(rows_with_all_elements_in_list)]
    return (cs_df, "CS chats filtered successfully!", True) if not cs_df.empty else (None, "No CS chats found.", False)


def sales_split(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str, bool]:
    if "outgoing_sender_ids" not in df.columns:
        return None, "Missing column 'outgoing_sender_ids'", False

    parse_column_list(df, "outgoing_sender_ids")
    sales_df = df[df["outgoing_sender_ids"].apply(rows_with_all_elements_not_in_list)]
    return (sales_df, "Sales chats filtered successfully!", True) if not sales_df.empty else (None, "No Sales chats found.", False)


# ----------------------------------------------------------------------
# 5) SIMPLE SEARCH / FILTER  (left as‑is)
# ----------------------------------------------------------------------
def search_messages(df: pd.DataFrame, text_column: str, searched_text: str) -> Tuple[Optional[pd.DataFrame], str]:
    if text_column not in df.columns:
        return None, f"Column '{text_column}' not found.",

    hits = df[df[text_column].str.contains(searched_text, case=False, na=False)]
    return (hits, "Search complete.") if not hits.empty else (None, "No hits found.")


def filter_by_chat_id(df: pd.DataFrame, chat_id_input: str) -> Tuple[Optional[pd.DataFrame], str, bool]:
    if "Chat ID" not in df.columns:
        return None, "Missing 'Chat ID' column.", False

    chat_id_input = str(chat_id_input).strip()
    out = df[df["Chat ID"].astype(str).str.strip() == chat_id_input]
    return (out, f"{len(out)} rows matched.", True) if not out.empty else (None, "No match.", False)


# ----------------------------------------------------------------------
# 6) MAKE GPT‑READABLE  (consume *paired* dataframe)
# ----------------------------------------------------------------------
def _parse_dates(lst):
    return [pd.to_datetime(x, errors="coerce") for x in lst]


def make_readable(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Take the dataframe returned by `pair_messages` and build a plain‑text
    transcript ordered chronologically, alternating Client / Agent lines.
    """
    required = {
        "Chat ID",
        "incoming_texts", "outgoing_texts",
        "incoming_dates", "outgoing_dates",
    }
    if not required.issubset(df.columns):
        return None, f"Expected paired dataframe with columns {required}"

    lines: List[str] = []

    for chat_id, grp in df.groupby("Chat ID"):
        timeline: List[Tuple[pd.Timestamp, str, str]] = []

        for _, row in grp.iterrows():
            in_texts = _safe_eval_list(row["incoming_texts"])
            out_texts = _safe_eval_list(row["outgoing_texts"])
            in_dates = _parse_dates(_safe_eval_list(row["incoming_dates"]))
            out_dates = _parse_dates(_safe_eval_list(row["outgoing_dates"]))

            timeline += list(zip(in_dates, ["Client"] * len(in_texts), in_texts))
            timeline += list(zip(out_dates, ["Agent"] * len(out_texts), out_texts))

        # sort; NaT (failed date parse) pushed to bottom
        timeline.sort(key=lambda x: (pd.isna(x[0]), x[0]))

        lines.append(f"Chat ID: {chat_id}")
        for _, speaker, msg in timeline:
            lines.append(f"{speaker}: {msg}")
        lines.append("-" * 70)
        lines.append("")

    transcript = "\n".join(lines)
    return transcript, "Transcript built successfully."
