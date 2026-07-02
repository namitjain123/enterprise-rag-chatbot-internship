import logfire
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.routes import router
from app.core.config import GUARDRAILS_ENABLED

logfire.configure(
    service_name="enterprise-rag-guardrails",
    send_to_logfire="if-token-present",
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Warm up the NeMo singleton at startup so the first user request isn't slow.
    if GUARDRAILS_ENABLED:
        with logfire.span("startup_warmup"):
            from app.guardrails.nemo_pipeline import get_rails
            get_rails()
            logfire.info("startup_warmup: NeMo rails ready")
    yield


app = FastAPI(title="Enterprise RAG Chatbot", lifespan=lifespan)
logfire.instrument_fastapi(app)

app.include_router(router)
