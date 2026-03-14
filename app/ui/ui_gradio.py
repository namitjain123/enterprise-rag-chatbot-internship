import gradio as gr
import requests

API_BASE = "http://127.0.0.1:8000"


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
        outputs=[user_input, chatbot]
    )

if __name__ == "__main__":
    demo.launch()