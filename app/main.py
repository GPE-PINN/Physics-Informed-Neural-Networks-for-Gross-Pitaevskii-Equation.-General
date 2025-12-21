from fastapi import FastAPI, Request
from app.forward.router import router as forward_router
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from fastapi import Request
from contextlib import asynccontextmanager
from app.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(forward_router)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc):
    return PlainTextResponse("bad request", status_code=400)

