from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.upload_pdfs import router as upload_router
from routes.ask_question import router as ask_router

app = FastAPI(title = "Medical Assistant API", description="API for AI Medical Assistant Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
from middlewares.exception_handlers import catch_exception_middleware
#middleware Exception Handler
app.middleware("http")(catch_exception_middleware)

# upload pdf documents
app.include_router(upload_router)

#Ask Query
app.include_router(ask_router)
