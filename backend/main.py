from fastapi import FastAPI
from backend.api.run import router as run_task_router
from backend.api.compare import router as compare_router
from backend.api.leaderboard import router as leaderboard_router

app = FastAPI(
    title="RAG Evaluation Platform",
    version="1.0.0"
)

# -------------------------
# Register routes
# -------------------------
app.include_router(run_task_router, prefix="/run-task")
app.include_router(compare_router, prefix="/compare")
app.include_router(leaderboard_router, prefix="/leaderboard")


# -------------------------
# Health check
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}