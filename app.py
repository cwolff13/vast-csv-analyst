"""Small Streamlit UI for chatting with workspace CSVs through PandasAI."""

import re
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import pandasai as pai
import streamlit as st
from pandasai_litellm.litellm import LiteLLM


WORKSPACE = Path("/workspace")
CHARTS = WORKSPACE / "charts"
VLLM_URL = "http://127.0.0.1:18000/v1"
MODEL = "Qwen/Qwen3.8-27B"

MODES = {
    "Fast": {
        "max_tokens": 1200,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "Deep": {
        "max_tokens": 4096,
        "reasoning_effort": "medium",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    },
}


def qwen_is_connected():
    try:
        urlopen(f"{VLLM_URL}/models", timeout=2).close()
        return True
    except OSError:
        return False


def make_agent(dataframe, mode):
    llm = LiteLLM(
        model=f"hosted_vllm/{MODEL}",
        api_base=VLLM_URL,
        **MODES[mode],
    )
    return pai.Agent(
        dataframe,
        config={"llm": llm, "max_retries": 1, "save_logs": True},
    )


def save_chart(response, question):
    CHARTS.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:50]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = CHARTS / f"{slug or 'chart'}-{timestamp}.png"
    response.save(str(path))
    return path


def previous_exchange(messages):
    for index in range(len(messages) - 1, -1, -1):
        answer = messages[index]
        if answer["role"] != "assistant" or answer["type"] == "error":
            continue

        for earlier in range(index - 1, -1, -1):
            if messages[earlier]["role"] == "user":
                value = (
                    "A chart was generated for this question."
                    if answer["type"] == "chart"
                    else str(answer["value"])
                )
                return messages[earlier]["value"], value
    return None


def follow_up_prompt(question, exchange):
    previous_question, previous_answer = exchange
    return (
        "Continue the conversation using the previous exchange below.\n\n"
        f"Previous question: {previous_question}\n"
        f"Previous answer: {previous_answer}\n\n"
        f"Current follow-up: {question}"
    )


def show_answer(message):
    if message["type"] == "chart":
        path = Path(message["value"])
        st.image(str(path), use_container_width=True)
        st.download_button(
            "Download chart",
            path.read_bytes(),
            file_name=path.name,
            mime="image/png",
            key=f"download-{path.name}",
        )
    elif message["type"] == "dataframe":
        st.dataframe(message["value"], use_container_width=True)
    elif message["type"] == "error":
        st.error(message["value"])
    else:
        st.write(message["value"])

    st.caption(f"Completed in {message['seconds']:.1f}s")
    if message.get("code"):
        with st.expander("Generated analysis code"):
            st.code(message["code"], language="python")


st.set_page_config(page_title="Vast CSV Analyst", page_icon="📊", layout="wide")
st.title("Vast CSV Analyst")
st.caption("Chat with a CSV using PandasAI and Qwen running on this Vast instance.")
st.caption(
    "Ask follow-up questions naturally. "
    "Use Deep mode for charts or more involved analysis."
)

csv_files = sorted(WORKSPACE.glob("*.csv"))
connected = qwen_is_connected()

with st.sidebar:
    st.header("Setup")
    if not csv_files:
        st.error("Add a CSV to /workspace, then refresh this page.")
        st.stop()

    selected_csv = st.selectbox(
        "Dataset",
        csv_files,
        format_func=lambda path: path.name,
    )
    mode = st.radio(
        "Mode",
        ["Fast", "Deep"],
        index=1,
        horizontal=True,
        help="Fast skips Qwen thinking. Deep is slower but better for harder analysis.",
    )

    if connected:
        st.success(f"Connected to {MODEL}")
    else:
        st.error("Qwen is not responding on port 18000.")

    new_chat = st.button("New conversation", use_container_width=True)

context = (str(selected_csv), mode)
if st.session_state.get("context") != context:
    with st.spinner(f"Loading {selected_csv.name}..."):
        st.session_state.dataframe = pai.read_csv(str(selected_csv))
        st.session_state.agent = make_agent(st.session_state.dataframe, mode)
    st.session_state.context = context
    st.session_state.messages = []

if new_chat:
    st.session_state.agent = make_agent(st.session_state.dataframe, mode)
    st.session_state.messages = []

dataframe = st.session_state.dataframe
st.caption(
    f"{selected_csv.name} · {len(dataframe):,} rows × "
    f"{len(dataframe.columns):,} columns · {mode} mode"
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["value"])
        else:
            show_answer(message)

question = st.chat_input("Ask a question about this dataset", disabled=not connected)
if question:
    st.session_state.messages.append({"role": "user", "value": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        started = time.perf_counter()
        with st.spinner(f"Qwen is analyzing in {mode} mode..."):
            try:
                exchange = previous_exchange(st.session_state.messages)
                prompt = (
                    follow_up_prompt(question, exchange) if exchange else question
                )
                agent = st.session_state.agent
                response = (
                    agent.follow_up(prompt) if exchange else agent.chat(prompt)
                )
                answer_type = response.type
                value = response.value
                if answer_type == "chart":
                    value = str(save_chart(response, question))
                answer = {
                    "role": "assistant",
                    "type": answer_type,
                    "value": value,
                    "seconds": time.perf_counter() - started,
                    "code": getattr(response, "last_code_executed", None),
                }
            except Exception as error:
                answer = {
                    "role": "assistant",
                    "type": "error",
                    "value": f"Analysis failed: {error}",
                    "seconds": time.perf_counter() - started,
                }

        st.session_state.messages.append(answer)
        show_answer(answer)
