"""FastAPI wrapper for the ai-quant-lab research loop.

Exposes a web UI and SSE endpoint so hypothesis generation, critic verdicts,
backtests, and gate results stream to the browser in real time.

Run:
    pip install fastapi uvicorn
    python -m wrapper
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from ai_quant_lab.agents.memory import ResearchMemory
from ai_quant_lab.backtest import BacktestConfig
from ai_quant_lab.orchestrator.loop import LoopConfig, run_research_loop

app = FastAPI(title="AI Quant Researcher")

# run_id -> {"status", "queue", "request", "started_at"}
_runs: dict[str, dict[str, Any]] = {}


class RunRequest(BaseModel):
    market_description: str = Field(..., min_length=5)
    market_type: str = "generic"
    iterations: int = Field(20, ge=1, le=200)
    target_survivors: int = Field(3, ge=1, le=20)
    cost_bps: float = Field(8.0, ge=0.0, le=100.0)
    ticker: str = Field("", description="If set, fetch real prices via defeatbeta. Leave blank for synthetic data.")


def _fetch_prices(ticker: str) -> pd.Series:
    from defeatbeta_api.data.ticker import Ticker  # noqa: PLC0415
    df = Ticker(ticker.upper()).price()
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.sort_values("report_date").set_index("report_date")
    return df["close"].rename(ticker.upper())


def _synthetic_prices(n_bars: int = 2520, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0004, 0.012, n_bars)
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    idx = pd.date_range("2015-01-01", periods=n_bars, freq="B")
    return pd.Series(prices, index=idx, name="close")


def _run_in_thread(run_id: str, request: RunRequest, q: "queue.Queue[dict]") -> None:
    def emit(event: dict) -> None:
        q.put(event)

    try:
        if request.ticker:
            price_data = _fetch_prices(request.ticker)
        else:
            price_data = _synthetic_prices()
        config = LoopConfig(
            market_description=request.market_description,
            market_type=request.market_type,
            iterations=request.iterations,
            target_survivors=request.target_survivors,
            backtest_config=BacktestConfig(cost_bps=request.cost_bps),
        )
        db_path = Path(f"./memory_{run_id}.db")
        with ResearchMemory(db_path) as memory:
            run_research_loop(
                price_data,
                config,
                memory=memory,
                log=lambda _: None,
                on_event=emit,
            )
    except Exception as exc:  # noqa: BLE001
        q.put({"type": "run_error", "message": str(exc)})
    finally:
        _runs[run_id]["status"] = "done"


@app.post("/api/run")
async def start_run(request: RunRequest) -> dict:
    run_id = uuid.uuid4().hex[:8]
    q: queue.Queue = queue.Queue()
    _runs[run_id] = {
        "status": "running",
        "queue": q,
        "request": request.model_dump(),
        "started_at": datetime.utcnow().isoformat(),
    }
    threading.Thread(target=_run_in_thread, args=(run_id, request, q), daemon=True).start()
    return {"run_id": run_id}


@app.get("/api/stream/{run_id}")
async def stream_run(run_id: str) -> StreamingResponse:
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")

    q = _runs[run_id]["queue"]

    async def generate():
        loop = asyncio.get_running_loop()

        def _get() -> dict | None:
            try:
                return q.get(timeout=25.0)
            except queue.Empty:
                return None

        while True:
            event = await loop.run_in_executor(None, _get)
            if event is None:
                yield ": keepalive\n\n"
                continue
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") in ("run_complete", "run_error"):
                break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
async def index() -> HTMLResponse:
    html = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html.read_text(encoding="utf-8"))
