import ast
import math
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

###############################################################################
#  CONFIGURATION                                                               #
###############################################################################

SALES_AGENT_IDS: List[int] = [389704, 268034, 146737, 129175, 383281, 472669, 238634]
IGNORED_CONTACT_IDS: List[int] = [21794581]

###############################################################################
#  LOW‑LEVEL HELPERS                                                           #
###############################################################################

def _safe_json(txt: str) -> Dict:
    try:
        return ast.literal_eval(txt) if isinstance(txt, str) else {}
    except Exception:
        return {}


def extract_text(content_json: str) -> str:
    """Return a readable surrogate for whatever is in *Content*."""
    data = _safe_json(content_json)
    mtype = data.get("type", "text")

    try:
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
    if isinstance(x, list):
        seq = x
    elif isinstance(x, str):
        try:
            seq = ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    else:
        return []

    out: List[int] = []
    for y in seq:
        try:
            if y is None or (isinstance(y, float) and math.isnan(y)):
                continue
            out.append(int(y))
        except Exception:
            continue
    return out

###############################################################################
#  PRE‑PROCESSING                                                              #
###############################################################################

def preprocess_dataframe(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
    REQUIRED = ["Contact ID", "Date & Time", "Message Type", "Sender ID", "Content"]
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        return None, f"Missing column(s): {', '.join(miss)}"

    df = df.copy()
    df = df[~df["Contact ID"].isin(IGNORED_CONTACT_IDS)]
    df["Date & Time"] = pd.to_datetime(df["Date & Time"], errors="coerce")
    df = df.dropna(subset=["Date & Time"]).sort_values(["Contact ID", "Date & Time"]).reset_index(drop=True)

    df["Chat ID"] = (df["Contact ID"] != df["Contact ID"].shift()).cumsum()
    df["text"] = df["Content"].apply(extract_text)

    for col in ("Type", "Sub Type", "Sender ID"):
        if col in df.columns:
            df[col] = df[col].fillna("unknown" if df[col].dtype == object else 0)

    lead = ["Chat ID", "Contact ID", "Date & Time", "Message Type", "Sender ID", "text", "Content"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    return df, "DataFrame pre‑processed successfully!"

###############################################################################
#  MESSAGE PAIRING                                                             #
###############################################################################

def _flush_pair(store: List[Dict], contact_id: int, chat_id: int,
                inc: List[Dict], out: List[Dict]):
    if not (inc and out):
        return  # skip incomplete pairs entirely
    store.append({
        "Chat ID": chat_id,
        "Contact ID": contact_id,
        "incoming_dates":  [m["Date & Time"] for m in inc],
        "outgoing_dates":  [m["Date & Time"] for m in out],
        "incoming_sender_ids": [m["Sender ID"] for m in inc],
        "outgoing_sender_ids": [m["Sender ID"] for m in out],
        "incoming_texts":  [m["text"] for m in inc],
        "outgoing_texts":  [m["text"] for m in out],
    })


def pair_messages(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
    if df is None or df.empty:
        return None, "No DataFrame provided."

    paired: List[Dict] = []

    for (contact_id, chat_id), grp in df.groupby(["Contact ID", "Chat ID"]):
        inc, out, buf = [], [], []
        side: Optional[str] = None  # current direction of *buf*

        for _, row in grp.iterrows():
            mtype = row["Message Type"].lower()
            if side is None:
                side = mtype
            if mtype != side:
                # dump buf into inc or out depending on previous side
                (inc if side == "incoming" else out).extend(buf)
                buf = [row]
                side = mtype
                if inc and out:
                    _flush_pair(paired, contact_id, chat_id, inc, out)
                    inc, out = [], []
            else:
                buf.append(row)

        # handle leftover buffer at end of chat
        if buf:
            (inc if side == "incoming" else out).extend(buf)
        _flush_pair(paired, contact_id, chat_id, inc, out)

    if not paired:
        return None, "No complete client/agent pairs produced."

    ordered_cols = [
        "Chat ID", "Contact ID",
        "incoming_dates", "outgoing_dates",
        "incoming_sender_ids", "outgoing_sender_ids",
        "incoming_texts", "outgoing_texts",
    ]
    return pd.DataFrame(paired)[ordered_cols], "Messages paired successfully!"

###############################################################################
#  SALES / CS SPLITS                                                           #
###############################################################################

def _all(lst: List[int], test) -> bool:
    return bool(lst) and all(test(x) for x in lst)


def sales_split(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str, bool]:
    if "outgoing_sender_ids" not in df.columns:
        return None, "Column 'outgoing_sender_ids' missing.", False
    df = df.copy()
    df["outgoing_sender_ids"] = df["outgoing_sender_ids"].apply(_as_int_list)
    mask = df["outgoing_sender_ids"].apply(lambda ids: _all(ids, lambda i: i in SALES_AGENT_IDS))
    sub = df[mask]
    return (sub, "Sales chats filtered successfully!", True) if not sub.empty else (None, "No Sales chats found.", False)


def cs_split(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str, bool]:
    if "outgoing_sender_ids" not in df.columns:
        return None, "Column 'outgoing_sender_ids' missing.", False
    df = df.copy()
    df["outgoing_sender_ids"] = df["outgoing_sender_ids"].apply(_as_int_list)
    mask = df["outgoing_sender_ids"].apply(lambda ids: _all(ids, lambda i: i not in SALES_AGENT_IDS))
    sub = df[mask]
    return (sub, "CS chats filtered successfully!", True) if not sub.empty else (None, "No CS chats found.", False)

###############################################################################
#  SEARCH, FILTER, READABLE                                                    #
###############################################################################

def search_messages(df: pd.DataFrame, column: str, phrase: str) -> Tuple[Optional[pd.DataFrame], str]:
    if column not in df.columns:
        return None, f"Column '{column}' missing."
    hits = df[df[column].str.contains(phrase, case=False, na=False)]
    return (hits, "Search completed.") if not hits.empty else (None, "No matches found.")


def filter_by_chat_id(df: pd.DataFrame, chat_id: str) -> Tuple[Optional[pd.DataFrame], str, bool]:
    if "Chat ID" not in df.columns:
        return None, "Column 'Chat ID' missing.", False
    chat_id = str(chat_id).strip()
    sub = df[df["Chat ID"].astype(str).str.strip() == chat_id]
    return (sub, f"{len(sub)} rows with Chat ID {chat_id}.", True) if not sub.empty else (None, "No such Chat ID.", False)


def make_readable(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    if df is None or df.empty:
        return None, "DataFrame empty."

    blocks: List[str] = []
    for (cid, chat_id), grp in df.groupby(["Contact ID", "Chat ID"]):
        blocks.append(f"Contact ID: {cid}\nChat ID: {chat_id}\n")
        for _, row in grp.iterrows():
            inc = row["incoming_texts"]
            out = row["outgoing_texts"]
            if isinstance(inc, str):
                inc = ast.literal_eval(inc)
            if isinstance(out, str):
                out = ast.literal_eval(out)
            blocks.extend([f"Client: {txt}" for txt in inc])
            blocks.extend([f"Agent: {txt}" for txt in out])
        blocks.append("\n" + "-" * 70 + "\n")

    transcript = "\n".join(blocks)
    with open("chat_transcript.txt", "w", encoding="utf-8") as fh:
        fh.write(transcript)
    return transcript, "Transcript saved to chat_transcript.txt"
