from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Enterprise RAG Chatbot")

app.include_router(router)