import gradio as gr
import requests

API_BASE = "http://127.0.0.1:8000"


def handle_upload(file):
    try:
        with open(file.name, "rb") as f:
            files = {"file": (file.name, f, "application/pdf")}
            response = requests.post(f"{API_BASE}/ingest", files=files)

        if response.status_code == 200:
            data = response.json()
            return f"{data['message']} Chunks indexed: {data['chunks_indexed']}"
        return f"Error: {response.text}"

    except Exception as e:
        return f"Error processing document: {str(e)}"


def respond(message, chat_history):
    if chat_history is None:
        chat_history = []

    if not message or not message.strip():
        return "", chat_history

    try:
        response = requests.post(
            f"{API_BASE}/query",
            json={"question": message, "top_k": 3},
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer returned.")
        else:
            answer = f"Error: {response.text}"

        # Messages format required by your Gradio version
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": answer})

        return "", chat_history

    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return "", chat_history


with gr.Blocks() as demo:
    gr.Markdown("## Enterprise RAG Chatbot")

    with gr.Row():
        file_upload = gr.File(label="Upload PDF Document", file_types=[".pdf"])
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
        outputs=[user_input, chatbot]
    )

if __name__ == "__main__":
    demo.launch()