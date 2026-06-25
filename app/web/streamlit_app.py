import os
from uuid import uuid4

import requests
import streamlit as st

from app.models import GetMessageRequestModel, IncomingMessage

DEFAULT_BOT_URL = os.getenv("FASTAPI_URL", "http://localhost:6872")

st.set_page_config(page_title="Echo Bot", initial_sidebar_state="expanded")
st.markdown("# Echo bot 🚀")

# ── session state init ──────────────────────────────────────────────────────
if "dialog_id" not in st.session_state:
    st.session_state.dialog_id = str(uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Type something", "probability": None}]
if "probabilities" not in st.session_state:
    st.session_state.probabilities = []  # list of floats from /predict responses

# ── sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("Reset"):
        st.session_state.messages = [{"role": "assistant", "content": "Type something", "probability": None}]
        st.session_state.probabilities = []
        st.session_state.dialog_id = str(uuid4())

    echo_bot_url = st.text_input("Bot URL", value=DEFAULT_BOT_URL, disabled=True)
    st.text_input("Dialog ID", value=st.session_state.dialog_id, disabled=True)

    st.markdown("---")
    st.markdown("### Metrics")
    probs = st.session_state.probabilities
    if probs:
        avg = sum(probs) / len(probs)
        last = probs[-1]
        st.metric("Avg bot probability", f"{avg:.1%}")
        st.metric("Last prediction", f"{last:.1%}")
        st.metric("Messages classified", len(probs))
    else:
        st.info("No predictions yet.")


def _prob_badge(prob: float | None) -> str:
    if prob is None:
        return ""
    pct = f"{prob:.0%}"
    color = "red" if prob > 0.6 else ("orange" if prob > 0.4 else "green")
    return f" :{color}[🤖 {pct}]"


def _call_predict(text: str, participant_index: int, msg_id: str) -> float | None:
    try:
        payload = IncomingMessage(
            text=text,
            dialog_id=st.session_state.dialog_id,
            id=msg_id,
            participant_index=participant_index,
        ).model_dump()
        payload = {k: str(v) if not isinstance(v, (str, int, float)) else v for k, v in payload.items()}
        r = requests.post(echo_bot_url + "/predict", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["is_bot_probability"]
    except Exception as exc:
        st.warning(f"Prediction failed: {exc}")
        return None


# ── render history ───────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    label = msg["content"] + _prob_badge(msg.get("probability"))
    st.chat_message(msg["role"]).write(label)

# ── handle new input ─────────────────────────────────────────────────────────
if user_text := st.chat_input():
    user_msg_id = str(uuid4())

    # 1. Classify user message (participant_index=0)
    user_prob = _call_predict(user_text, participant_index=0, msg_id=user_msg_id)
    if user_prob is not None:
        st.session_state.probabilities.append(user_prob)

    user_entry = {"role": "user", "content": user_text, "probability": user_prob}
    st.session_state.messages.append(user_entry)
    st.chat_message("user").write(user_text + _prob_badge(user_prob))

    # 2. Echo from bot (participant_index=1, same text — used to enrich dialog context)
    bot_msg_id = str(uuid4())
    echo_text = user_text  # echo bot always mirrors the user

    try:
        resp = requests.post(
            echo_bot_url + "/get_message",
            json=GetMessageRequestModel(
                dialog_id=st.session_state.dialog_id,
                last_msg_text=user_text,
                last_message_id=uuid4(),
            ).model_dump(),
            timeout=10,
        )
        echo_text = resp.json().get("new_msg_text", user_text)
    except Exception:
        pass

    # 3. Classify bot echo (participant_index=1); since echo == user text,
    #    this second call gives the model the full 2-turn context, improving accuracy.
    bot_prob = _call_predict(echo_text, participant_index=1, msg_id=bot_msg_id)
    if bot_prob is not None:
        st.session_state.probabilities.append(bot_prob)

    bot_entry = {"role": "assistant", "content": echo_text, "probability": bot_prob}
    st.session_state.messages.append(bot_entry)
    st.chat_message("assistant").write(echo_text + _prob_badge(bot_prob))

    st.rerun()
