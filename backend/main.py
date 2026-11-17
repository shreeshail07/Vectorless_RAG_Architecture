
from fastapi import FastAPI

from backend.api.health import router as health_router

app = FastAPI(
    title="Insurance Claims Intelligence API",
    description="Enterprise Insurance Claims Intelligence Platform powered by Vectorless RAG",
    version="1.0.0",
)

# Register API routers
app.include_router(health_router)


@app.get("/")
def root():
    return {
        "message": "Welcome to Insurance Claims Intelligence API",
        "documentation": "/docs",
    }
