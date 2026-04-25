from fastapi import FastAPI
from backend.api.run import router as run_task_router
from backend.api.compare import router as compare_router
from backend.api.leaderboard import router as leaderboard_router
from backend.api.experiments import router as experiments_router
from backend.api.recommend import router as recommend_router

app = FastAPI(
    title="LLM Evaluation Platform",
    version="1.0.0"
)

# -------------------------
# Register routes
# -------------------------
app.include_router(run_task_router, prefix="/run-task")
app.include_router(compare_router, prefix="/compare")
app.include_router(leaderboard_router, prefix="/leaderboard")
app.include_router(experiments_router, prefix="/experiments")
app.include_router(recommend_router, prefix="/recommend")

# -------------------------
# Health check
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Welcome to the LLM Evaluation Platform API!"}