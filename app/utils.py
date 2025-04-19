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

def pair_messages(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Collect consecutive incoming / outgoing runs per Chat‑ID while preserving
    exact timestamp order.  Output columns are identical to the old version
    so downstream tasks keep working.
    """
    try:
        if df is None or df.empty:
            raise ValueError("No DataFrame loaded.")

        required = {
            "Chat ID", "Contact ID", "Message Type",
            "Date & Time", "Sender ID", "text"
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = df.sort_values(["Chat ID", "Date & Time"])

        rows = []
        for chat_id, chat in df.groupby("Chat ID"):
            cur_dir = None
            buf_in, buf_out = [], []
            for _, row in chat.iterrows():
                if row["Message Type"] == cur_dir or cur_dir is None:
                    # keep filling the current run
                    if row["Message Type"] == "incoming":
                        buf_in.append(row)
                    else:
                        buf_out.append(row)
                    cur_dir = row["Message Type"]
                else:
                    # direction changed → flush a paired row
                    rows.append({
                        "Chat ID": chat_id,
                        "Contact ID": row["Contact ID"],
                        "incoming_dates": [r["Date & Time"] for r in buf_in],
                        "outgoing_dates": [r["Date & Time"] for r in buf_out],
                        "incoming_sender_ids": [r["Sender ID"] for r in buf_in],
                        "outgoing_sender_ids": [r["Sender ID"] for r in buf_out],
                        "incoming_texts": [r["text"] for r in buf_in],
                        "outgoing_texts": [r["text"] for r in buf_out],
                    })
                    # reset buffers with the first row of new run
                    buf_in, buf_out = ([], [row]) if row["Message Type"] == "outgoing" else ([row], [])
                    cur_dir = row["Message Type"]

            # flush whatever is left at end of chat
            rows.append({
                "Chat ID": chat_id,
                "Contact ID": chat.iloc[-1]["Contact ID"],
                "incoming_dates": [r["Date & Time"] for r in buf_in],
                "outgoing_dates": [r["Date & Time"] for r in buf_out],
                "incoming_sender_ids": [r["Sender ID"] for r in buf_in],
                "outgoing_sender_ids": [r["Sender ID"] for r in buf_out],
                "incoming_texts": [r["text"] for r in buf_in],
                "outgoing_texts": [r["text"] for r in buf_out],
            })

        paired_df = pd.DataFrame(rows)
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

# utils.py  ── replace ONLY make_readable ------------------------------
import pandas as pd
import ast
from typing import Tuple, Optional, List

def _as_list(x):
    """Turn a stringified list into a real list, tolerate NaNs."""
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        out = ast.literal_eval(x)
        return out if isinstance(out, list) else []
    except Exception:
        return []

def make_readable(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Build a transcript from the *paired* dataframe so that every
    Client message is followed immediately by the matching Agent reply
    (if there is one), preserving the turn order inside each Chat ID.
    Place‑holders like 'template', 'image', etc. are kept.
    """
    needed = {
        "Chat ID",
        "incoming_texts", "outgoing_texts"
    }
    if not needed.issubset(df.columns):
        return None, f"Expected paired dataframe with columns {needed}"

    lines: List[str] = []

    for chat_id, grp in df.groupby("Chat ID"):
        lines.append(f"Chat ID: {chat_id}")

        # the paired rows are already sequential turns
        for _, row in grp.iterrows():
            ins  = _as_list(row["incoming_texts"])
            outs = _as_list(row["outgoing_texts"])

            # interleave  -----------------------------------------------------
            for i in range(max(len(ins), len(outs))):
                if i < len(ins):
                    lines.append(f"Client: {ins[i]}")
                if i < len(outs):
                    lines.append(f"Agent: {outs[i]}")

        lines.append("-" * 70)
        lines.append("")                 # blank line

    transcript = "\n".join(lines)
    return transcript, "Transcript built successfully (interleaved)."
def make_readable(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Build a transcript from the *paired* dataframe, emitting the next message
    in chronological order – client or agent – so the flow is:
        Client msg 1
        Agent  reply 1
        Client msg 2
        Agent  reply 2
        ...
    """
    import ast, pandas as pd
    from typing import List, Tuple

    must_have = {
        "Chat ID",
        "incoming_texts", "outgoing_texts",
        "incoming_dates", "outgoing_dates",
    }
    if not must_have.issubset(df.columns):
        return None, f"Expected paired dataframe with columns {must_have}"

    def _l(x):  # safe list loader
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        try:
            out = ast.literal_eval(x)
            return out if isinstance(out, list) else []
        except Exception:
            return []

    def _d(dlst):
        return [pd.to_datetime(v, errors="coerce") for v in dlst]

    lines: List[str] = []

    for chat_id, grp in df.groupby("Chat ID"):
        # flatten the chat into (time, speaker, text)
        timeline: List[Tuple[pd.Timestamp, str, str]] = []
        for _, row in grp.iterrows():
            in_txt, out_txt = _l(row["incoming_texts"]), _l(row["outgoing_texts"])
            in_dt,  out_dt  = _d(_l(row["incoming_dates"])), _d(_l(row["outgoing_dates"]))
            timeline.extend(zip(in_dt,  ["Client"]*len(in_txt), in_txt))
            timeline.extend(zip(out_dt, ["Agent"]*len(out_txt),  out_txt))

        # order by timestamp; NaTs go last but keep insertion order there
        timeline.sort(key=lambda x: (pd.isna(x[0]), x[0]))

        # emit
        lines.append(f"Chat ID: {chat_id}")
        for _, speaker, text in timeline:
            lines.append(f"{speaker}: {text}")
        lines.append("-"*70)
        lines.append("")

    transcript = "\n".join(lines)
    return transcript, "Transcript built successfully."
