"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .api.routes import router
from .database import Base, SessionLocal, engine, run_lightweight_migrations
from .services.pipeline import ApiError, seed_default_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    run_lightweight_migrations(engine)
    db = SessionLocal()
    try:
        seed_default_model(db)
    finally:
        db.close()
    yield


app = FastAPI(
    title="Seismic Network Phase Association API",
    version="1.9.0",
    description=(
        "Upload miniSEED waveform windows, generate STA/LTA candidate picks, "
        "associate them into events and locate them with weighted least squares "
        "against versioned 1-D layered velocity models. Per-station P/S arrival "
        "calibrations can be estimated from historical events and atomically "
        "activated; process/replay requests may then correct arrivals before "
        "location. A cross-run catalog deterministically merges the same "
        "earthquake's events from multiple completed runs into immutable, "
        "lifecycle-managed catalog entries (draft/published/withdrawn). A "
        "station impact audit freezes one completed event and re-runs "
        "association/location after removing each participating station's P/S "
        "picks, flagging the key stations. A catalog alert orchestration layer "
        "lets administrators publish immutable alert rule versions over the "
        "published catalog and evaluate catalog compilations into "
        "deterministic, mergeable alert events with a full audit chain. A "
        "P-wave first-motion focal mechanism inversion grid-searches "
        "double-couple strike/dip/rake on a 10-degree grid for a completed "
        "event's associated P polarities, keeping every minimum-misfit "
        "mechanism with plane-swap duplicates removed but the "
        "slip-reversed (-M) mechanism retained as a separate candidate. An "
        "event location uncertainty assessment freezes a completed event's "
        "associated picks, weights, velocity model and calibration version, "
        "then bootstrap-relocates it on seeded with-replacement station "
        "resamples, reporting nearest-rank confidence intervals, the "
        "solvable-only sample covariance and the solvable rate. A station "
        "P-wave pick gap assessment takes a catalog compilation id plus an "
        "arrival tolerance and minimum observable event count, considers only "
        "the entries already published at submission (each once via its master "
        "run/event), predicts P arrivals with that run's velocity model and "
        "station-P calibration offset, requires the ingested waveform to fully "
        "cover the predicted instant +/- tolerance, then marks manual-P or "
        "STA/LTA matches as hits (manual S never matches) and the rest as "
        "gaps, reporting per-station observable/gap counts and gap rates "
        "(with evidence-insufficient flags) plus paginated per-event evidence. "
        "A station waveform acquisition continuity check reads one station's "
        "waveform metadata across all ingestion batches for a half-open UTC "
        "window, clips each segment (start + npts/sampling-rate) to the "
        "window, counts identical segments (same start, sampling rate, npts "
        "and data hash) once while retaining every ingestion id, then sweeps "
        "the endpoints and reports merged covered segments, leading/trailing/"
        "internal gaps and overlapping acquisition ranges with their source "
        "ingestion ids; overlap means range coincidence, never sample-content "
        "conflict."
    ),
    lifespan=lifespan,
)
app.include_router(router)


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": {"code": exc.code, "message": exc.message}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = [
        {"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": "validation_error",
                "message": "request failed schema validation",
                "errors": errors,
            }
        },
    )


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "seismic-network-association",
        "version": "1.9.0",
        "docs": "/docs",
        "health": "/health",
    }
