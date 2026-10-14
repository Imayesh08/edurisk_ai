#!/bin/bash
# Convenient local runner for EduRisk AI (FastAPI + Streamlit)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Ensure venv is active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f ".venv/bin/activate" ]; then
        echo "[INFO] Activating .venv virtual environment..."
        source .venv/bin/activate
    else
        echo "[ERROR] .venv not found. Please create it or run pip install -r requirements.txt"
        exit 1
    fi
fi

echo "=========================================================="
echo "  EduRisk AI — Launching Local Services"
echo "=========================================================="
echo "1. Starting FastAPI Backend on http://127.0.0.1:8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "2. Starting Streamlit Dashboard on http://localhost:8501..."
streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0 &
STREAMLIT_PID=$!

echo ""
echo "Services are now running!"
echo " - Streamlit Dashboard: http://localhost:8501"
echo " - FastAPI Swagger UI:  http://127.0.0.1:8000/docs"
echo ""
echo "Press Ctrl+C to stop both services."

# Trap SIGINT to gracefully terminate both background processes
trap "echo 'Stopping services...'; kill $API_PID $STREAMLIT_PID; exit 0" SIGINT SIGTERM
wait
