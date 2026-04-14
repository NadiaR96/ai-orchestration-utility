from fastapi import FastAPI
from backend.api.run_task import router as run_task_router
from backend.api.compare import router as compare_router

app = FastAPI(
    title="RAG Evaluation Platform",
    version="1.0.0"
)

# -------------------------
# Register routes
# -------------------------
app.include_router(run_task_router, prefix="/run-task")
app.include_router(compare_router, prefix="/compare")


# -------------------------
# Health check
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}