import ast
import math
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

###############################################################################
#  CONFIGURATION                                                               #
###############################################################################

# 1️⃣  IDs of agents that belong to the SALES team.
SALES_AGENT_IDS: List[int] = [389704, 268034, 146737, 129175, 383281, 472669, 238634]

# 2️⃣  IDs of contacts that must be *ignored* completely (e.g. test numbers).
IGNORED_CONTACT_IDS: List[int] = [21794581]

###############################################################################
#  LOW-LEVEL HELPERS                                                           #
###############################################################################

def _safe_json(content: str) -> Dict:
    """Return a dict parsed from *content* or an empty dict on failure."""
    try:
        return ast.literal_eval(content) if isinstance(content, str) else {}
    except Exception:
        return {}


def extract_text(content_json: str) -> str:
    """Extract a human-readable payload from the *Content* column.

    Rules
    -----
    * If the message type is **text**, return the actual text.
    * If the message type is **attachment**, return ``attachment (<type>)``.
    * If the message type is **whatsapp_template**, return ``template (<category>)``.
    * Otherwise fall back to ``other (<type>)`` or *unreadable* if parsing dies.
    """
    data = _safe_json(content_json)

    try:
        mtype = data.get("type", "text")
        if mtype == "text":
            return data.get("text", "text (empty)")
        if mtype == "attachment":
            atype = data.get("attachment", {}).get("type", "unknown")
            return f"attachment ({atype})"
        if mtype == "whatsapp_template":
            cat = data.get("template", {}).get("category", "unknown")
            return f"template ({cat})"
        return f"other ({mtype})"
    except Exception:
        return "unreadable"


def _as_int_list(x) -> List[int]:
    """Parse *x* (stringified list or list) into *List[int]*, NaNs removed."""
    if isinstance(x, list):
        seq = x
    elif isinstance(x, str):
        try:
            seq = ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    else:
        return []

    result: List[int] = []
    for elem in seq:
        try:
            if elem is None or (isinstance(elem, float) and math.isnan(elem)):
                continue
            result.append(int(elem))
        except Exception:
            continue
    return result

###############################################################################
#  PRE-PROCESSING                                                              #
###############################################################################

def preprocess_dataframe(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
    """Clean raw export and add helper columns.

    Steps
    -----
    1.  Ensure mandatory columns exist.
    2.  Drop rows with *Contact ID* in ``IGNORED_CONTACT_IDS``.
    3.  Convert ``Date & Time`` to **UTC-aware** pandas ``datetime``.
    4.  Sort by *(Contact ID, Date & Time)*.
    5.  Build an integer *Chat ID* that increments when ``Contact ID`` changes.
    6.  Generate ``text`` column via :func:`extract_text`.
    7.  Standardise nulls in a few noisy columns.
    8.  Return a trimmed, ordered DataFrame ready for pairing.
    """
    REQUIRED = [
        "Contact ID",
        "Date & Time",
        "Message Type",  # "incoming" / "outgoing"
        "Sender ID",
        "Content",
    ]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return None, f"Missing column(s): {', '.join(missing)}"

    # Core cleaning
    df = df.copy()
    df = df[~df["Contact ID"].isin(IGNORED_CONTACT_IDS)]
    df["Date & Time"] = pd.to_datetime(df["Date & Time"], errors="coerce")
    df = df.dropna(subset=["Date & Time"]).sort_values(["Contact ID", "Date & Time"]).reset_index(drop=True)

    # Chat identifier (per contact thread)
    df["Chat ID"] = (df["Contact ID"] != df["Contact ID"].shift()).cumsum()

    # Human-readable text column
    df["text"] = df["Content"].apply(extract_text)

    # Fill obvious nulls
    filler_cols = ["Type", "Sub Type", "Sender ID"]
    for col in filler_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown" if df[col].dtype == object else 0)

    # Re-order columns: Chat ID, Contact ID, Date & Time … rest
    lead_cols = ["Chat ID", "Contact ID", "Date & Time", "Message Type", "Sender ID", "text", "Content"]
    remaining = [c for c in df.columns if c not in lead_cols]
    df = df[lead_cols + remaining]

    return df, "DataFrame pre-processed successfully!"

###############################################################################
#  PAIR MESSAGES                                                               #
###############################################################################

def pair_messages(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
    """Aggregate consecutive *incoming* and *outgoing* runs into dialogue turns.

    Each row of the output contains **one** block of client text followed by the
    subsequent block of agent text (or vice-versa).  We deliberately keep lists
    (dates, sender IDs, texts) so that no granularity is lost.
    """
    if df is None or df.empty:
        return None, "No DataFrame provided."

    paired_rows: List[Dict] = []

    current_contact: Optional[int] = None
    buffer: List[Dict] = []          # accumulation of consecutive messages
    direction: Optional[str] = None  # "incoming" or "outgoing"

    def _flush(contact_id: int, chat_id: int, inc: List[Dict], out: List[Dict]):
        if not inc and not out:
            return
        paired_rows.append({
            "Chat ID": chat_id,
            "Contact ID": contact_id,
            "incoming_dates": [m["Date & Time"] for m in inc],
            "outgoing_dates": [m["Date & Time"] for m in out],
            "incoming_sender_ids": [m["Sender ID"] for m in inc],
            "outgoing_sender_ids": [m["Sender ID"] for m in out],
            "incoming_texts": [m["text"] for m in inc],
            "outgoing_texts": [m["text"] for m in out],
        })

    # --- main loop ---------------------------------------------------------
    inc_msgs: List[Dict] = []
    out_msgs: List[Dict] = []

    for _, row in df.iterrows():
        contact_id = row["Contact ID"]
        chat_id = row["Chat ID"]
        mtype = row["Message Type"].lower()

        if contact_id != current_contact:
            # new conversation → flush leftovers for previous contact
            _flush(current_contact, chat_id, inc_msgs, out_msgs)
            inc_msgs, out_msgs = [], []
            direction = None
            current_contact = contact_id

        # direction switch? push current buffer
        if direction is None:
            direction = mtype
        elif mtype != direction:
            # move accumulated buffer to the correct pile
            if direction == "incoming":
                inc_msgs.extend(buffer)
            else:
                out_msgs.extend(buffer)
            buffer = []
            direction = mtype

            # if we now have both sides, flush and reset for next turn
            if inc_msgs and out_msgs:
                _flush(contact_id, chat_id, inc_msgs, out_msgs)
                inc_msgs, out_msgs = [], []

        buffer.append(row)

    # handle tail
    if buffer:
        if direction == "incoming":
            inc_msgs.extend(buffer)
        else:
            out_msgs.extend(buffer)
    _flush(current_contact, df.iloc[-1]["Chat ID"], inc_msgs, out_msgs)

    if not paired_rows:
        return None, "No message pairs produced."

    paired_df = pd.DataFrame(paired_rows)[[
        "Chat ID",
        "Contact ID",
        "incoming_dates",
        "outgoing_dates",
        "incoming_sender_ids",
        "outgoing_sender_ids",
        "incoming_texts",
        "outgoing_texts",
    ]]

    return paired_df, "Messages paired successfully!"

###############################################################################
#  SALES ↔ CS SPLITS                                                           #
###############################################################################

def _every(lst: List[int], predicate) -> bool:
    return bool(lst) and all(predicate(x) for x in lst)


def cs_split(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str, bool]:
    """Return rows whose *outgoing_sender_ids* belong **exclusively** to CS."""
    if "outgoing_sender_ids" not in df.columns:
        return None, "Column 'outgoing_sender_ids' missing.", False

    df = df.copy()
    df["outgoing_sender_ids"] = df["outgoing_sender_ids"].apply(_as_int_list)
    mask = df["outgoing_sender_ids"].apply(
        lambda ids: _every(ids, lambda i: i not in SALES_AGENT_IDS)
    )
    subset = df[mask]
    return (subset, "CS chats filtered successfully!", True) if not subset.empty else (None, "No CS chats found.", False)


def sales_split(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str, bool]:
    """Return rows whose *outgoing_sender_ids* belong **exclusively** to SALES."""
    if "outgoing_sender_ids" not in df.columns:
        return None, "Column 'outgoing_sender_ids' missing.", False

    df = df.copy()
    df["outgoing_sender_ids"] = df["outgoing_sender_ids"].apply(_as_int_list)
    mask = df["outgoing_sender_ids"].apply(
        lambda ids: _every(ids, lambda i: i in SALES_AGENT_IDS)
    )
    subset = df[mask]
    return (subset, "Sales chats filtered successfully!", True) if not subset.empty else (None, "No Sales chats found.", False)

###############################################################################
#  SEARCH & PRETTY PRINT                                                      #
###############################################################################

def search_messages(df: pd.DataFrame, column: str, text: str) -> Tuple[Optional[pd.DataFrame], str]:
    if column not in df.columns:
        return None, f"Column '{column}' missing."
    hits = df[df[column].str.contains(text, case=False, na=False)]
    return (hits, "Search completed.") if not hits.empty else (None, "No matches found.")


def filter_by_chat_id(df: pd.DataFrame, chat_id: str) -> Tuple[Optional[pd.DataFrame], str, bool]:
    col = "Chat ID"
    if col not in df.columns:
        return None, f"Column '{col}' missing.", False
    chat_id = str(chat_id).strip()
    sub = df[df[col].astype(str).str.strip() == chat_id]
    return (sub, f"{len(sub)} rows with Chat ID {chat_id}.", True) if not sub.empty else (None, "No such Chat ID.", False)


def make_readable(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """Convert *paired* DataFrame to a plain-text transcript."""
    if df is None or df.empty:
        return None, "DataFrame empty."

    lines: List[str] = []

    for (contact_id, chat_id), grp in df.groupby(["Contact ID", "Chat ID"]):
        lines.append(f"Contact ID: {contact_id}\nChat ID: {chat_id}\n")
        inc: List[str] = []
        out: List[str] = []
        for _, row in grp.iterrows():
            inc.extend(_as_int_list(row["incoming_texts"]) if isinstance(row["incoming_texts"], str) else row["incoming_texts"])
            out.extend(_as_int_list(row["outgoing_texts"]) if isinstance(row["outgoing_texts"], str) else row["outgoing_texts"])
        lines.extend([f"Client: {txt}" for txt in inc])
        lines.extend([f"Agent: {txt}" for txt in out])
        lines.append("\n" + "-" * 70 + "\n")

    transcript = "\n".join(lines)
    with open("chat_transcript.txt", "w", encoding="utf-8") as fh:
        fh.write(transcript)

    return transcript, "Transcript saved to chat_transcript.txt"
