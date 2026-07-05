import os
import sys

# Allow running this file directly (`python app/ui/ui_gradio.py`): put the project
# root on sys.path so `import app...` resolves the same as `python -m`.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gradio as gr
import requests

# Use http://backend:8000 inside Docker; falls back to localhost when run directly.
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")


def handle_upload(files):
    try:
        if not files:
            return "No files uploaded."

        uploaded_count = 0
        total_chunks = 0

        for file in files:
            with open(file.name, "rb") as f:
                file_data = {"files": (file.name, f, "application/pdf")}
                response = requests.post(f"{API_BASE}/ingest", files=file_data)

            if response.status_code == 200:
                data = response.json()
                uploaded_count += 1
                total_chunks += data.get("chunks_indexed", 0)
            else:
                return f"Error uploading {file.name}: {response.text}"

        return f"{uploaded_count} documents uploaded successfully. Total chunks indexed: {total_chunks}"

    except Exception as e:
        return f"Error processing documents: {str(e)}"


def respond(message, chat_history):
    if chat_history is None:
        chat_history = []

    if not message or not message.strip():
        yield "", chat_history
        return

    # Append placeholder so the assistant bubble appears immediately.
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    yield "", chat_history

    try:
        with requests.get(
            f"{API_BASE}/query/stream",
            params={"question": message},
            stream=True,
            timeout=60,
        ) as r:
            for raw_line in r.iter_lines():
                if not raw_line:
                    continue
                # SSE lines look like: b"data: <token>"
                if raw_line.startswith(b"data: "):
                    token = raw_line[6:].decode("utf-8")
                    if token == "[DONE]":
                        break
                    chat_history[-1]["content"] += token
                    yield "", chat_history

    except Exception as e:
        chat_history[-1]["content"] = f"Error: {str(e)}"
        yield "", chat_history


with gr.Blocks() as demo:
    gr.Markdown("## Enterprise RAG Chatbot")

    with gr.Row():
        file_upload = gr.File(label="Upload PDF Documents", file_types=[".pdf"], file_count="multiple")
        upload_status = gr.Textbox(label="Upload Status", interactive=False)

    file_upload.change(
        fn=handle_upload,
        inputs=file_upload,
        outputs=upload_status
    )

    chatbot = gr.Chatbot(label="Chat History")
    user_input = gr.Textbox(
        placeholder="Ask a question about the uploaded document...",
        label="Your Question",
    )

    send_btn = gr.Button("Send")

    send_btn.click(
        fn=respond,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot],
    )
    user_input.submit(
        fn=respond,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot],
    )

# Chatbot + guardrails UI on port 7860.
# (RAGAS evaluation is a separate Streamlit app — run `streamlit run app/ui/eval_ui.py`
#  on port 8501. The two apps are intentionally kept independent.)
if __name__ == "__main__":
    demo.queue()   # required for generator-based streaming to work
    demo.launch(server_name="0.0.0.0", server_port=7860)