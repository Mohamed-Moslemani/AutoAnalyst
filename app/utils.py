# ===============================================
#  utils.py  (drop‑in replacement)
# ===============================================
import ast, json
from typing import Tuple, Optional, List, Dict
import pandas as pd
import numpy as np


# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------
CS_AGENT_IDS = {
    124760, 396575, 354259, 352740, 178283,
    398639, 467165, 277476, 464154, 1023356,
}


# ------------------------------------------------------------------
# 1)  TEXT NORMALISATION
# ------------------------------------------------------------------
def extract_text(content: str) -> str:
    """
    Convert the JSON blob in “Content” to plain text.
    Keeps 'template', 'image', etc. so you can filter later if desired.
    """
    try:
        obj = json.loads(content)
        typ = obj.get("type", "").lower()

        if typ == "text":
            return obj.get("text", "")

        if typ == "attachment":
            return obj.get("attachment", {}).get("type", "")

        if typ == "whatsapp_template":
            comps = obj.get("template", {}).get("components", [])
            texts = [c.get("text", "") for c in comps if "text" in c]
            return " ".join(texts) or "template"

        return "template"
    except (json.JSONDecodeError, TypeError):
        return "template"


# ------------------------------------------------------------------
# 2)  PRE‑PROCESS
# ------------------------------------------------------------------
def preprocess_dataframe(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
    """
    * parse dates
    * sort inside Contact ID
    * drop internal contact 21794581
    * add 'text' and incremental 'Chat ID'
    """
    try:
        if df is None or df.empty:
            raise ValueError("No data.")

        df["Date & Time"] = pd.to_datetime(df["Date & Time"], errors="coerce")
        df = df.sort_values(["Contact ID", "Date & Time"]).reset_index(drop=True)
        df = df[df["Contact ID"] != 21794581]

        df["text"]    = df["Content"].apply(extract_text)
        df["Chat ID"] = (df["Contact ID"] != df["Contact ID"].shift()).cumsum()

        # NA hygiene the app relied on
        df["text"].fillna("template", inplace=True)
        df["Type"].fillna("normal_text", inplace=True)
        df["Sub Type"].fillna("normal_text", inplace=True)
        df["Sender ID"].fillna(0, inplace=True)

        # nicer column order
        cols = df.columns.tolist()
        cols.insert(0, cols.pop(cols.index("Chat ID")))
        cols.insert(cols.index("Content") + 1, cols.pop(cols.index("text")))
        df = df[cols]

        return df, "Pre‑processing complete."
    except Exception as e:
        return None, f"Pre‑processing failed – {e}"


# ------------------------------------------------------------------
# 3)  PAIR  (keeps timestamp order inside each turn)
# ------------------------------------------------------------------
def pair_messages(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        if df is None or df.empty:
            raise ValueError("No data.")

        df = df.sort_values(["Chat ID", "Date & Time"])

        rows: List[Dict] = []
        for chat_id, chat in df.groupby("Chat ID"):
            cur_dir = None
            buf_in, buf_out = [], []

            def flush():
                if not buf_in and not buf_out:
                    return
                rows.append({
                    "Chat ID": chat_id,
                    "incoming_dates":  [r["Date & Time"] for r in buf_in],
                    "outgoing_dates":  [r["Date & Time"] for r in buf_out],
                    "incoming_sender_ids": [r["Sender ID"] for r in buf_in],
                    "outgoing_sender_ids": [r["Sender ID"] for r in buf_out],
                    "incoming_texts":  [r["text"] for r in buf_in],
                    "outgoing_texts":  [r["text"] for r in buf_out],
                })

            for _, row in chat.iterrows():
                direction = row["Message Type"]  # 'incoming' / 'outgoing'
                if cur_dir is None or direction == cur_dir:
                    (buf_in if direction == "incoming" else buf_out).append(row)
                    cur_dir = direction
                else:
                    flush()
                    buf_in, buf_out = ([], [row]) if direction == "outgoing" else ([row], [])
                    cur_dir = direction
            flush()

        out_df = pd.DataFrame(rows)
        return out_df, "Pairing complete."
    except Exception as e:
        return None, f"Pairing failed – {e}"


# ------------------------------------------------------------------
# 4)  SALES‑ONLY FILTER  (works on paired dataframe)
# ------------------------------------------------------------------
def sales_split(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str, bool]:
    if "outgoing_sender_ids" not in df.columns:
        return None, "Missing 'outgoing_sender_ids'.", False

    # normalise to real lists
    def to_list(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        try:
            val = ast.literal_eval(x)
            return val if isinstance(val, list) else []
        except Exception:
            return []

    df["outgoing_sender_ids"] = df["outgoing_sender_ids"].apply(to_list)

    is_sales = df["outgoing_sender_ids"].apply(
        lambda lst: all(s not in CS_AGENT_IDS for s in lst) or not lst
    )
    sales_df = df[is_sales].reset_index(drop=True)

    return (sales_df, "Sales chats isolated.", True) if not sales_df.empty else (None, "No sales chats found.", False)


# ------------------------------------------------------------------
# 5)  GPT‑READABLE TRANSCRIPT (strict chrono order)
# ------------------------------------------------------------------
def make_readable(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Build a txt transcript from the *Sales‑only paired* dataframe.
    True chronological order, speaker labels preserved.
    """
    need = {
        "Chat ID",
        "incoming_texts", "outgoing_texts",
        "incoming_dates", "outgoing_dates"
    }
    if not need.issubset(df.columns):
        return None, f"Expect columns {need}"

    def l(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        try:
            y = ast.literal_eval(x)
            return y if isinstance(y, list) else []
        except Exception:
            return []

    def d(lst):
        return [pd.to_datetime(v, errors="coerce") for v in lst]

    chunks = []
    for chat_id, grp in df.groupby("Chat ID"):
        timeline: List[Tuple[pd.Timestamp, str, str]] = []
        for _, row in grp.iterrows():
            timeline += list(zip(d(l(row["incoming_dates"])), ["Client"] * len(l(row["incoming_texts"])), l(row["incoming_texts"])))
            timeline += list(zip(d(l(row["outgoing_dates"])), ["Agent"]  * len(l(row["outgoing_texts"])),  l(row["outgoing_texts"])))
        timeline.sort(key=lambda t: (pd.isna(t[0]), t[0]))

        chunks.append(f"Chat ID: {chat_id}")
        for _, speaker, msg in timeline:
            chunks.append(f"{speaker}: {msg}")
        chunks.append("-" * 70)
        chunks.append("")

    txt = "\n".join(chunks)
    return txt, "Transcript ready."
