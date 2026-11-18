"""FastAPI application (runs on Laptop 1).

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    (or deployment/start_api.sh)

Endpoints
    GET  /health                              liveness + model/data status
    GET  /api/evaluation/trips                trip ids available on this server
    POST /api/predict  {"trip_id": "..."}     full pipeline; returns route + metrics
    GET  /api/evaluation/results/{trip_id}    metrics of the last prediction of that trip

Every request is logged (client address, method, path, status, duration) to the
console and to logs/api.log, together with each processing stage.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import EvaluationResult, HealthResponse, PredictRequest, PredictResponse, TripListResponse
from src.api.service import PredictionService, ServiceUnavailable, TripNotFound
from src.api.settings import EXPECTED_EVALUATION_TRIPS, Settings

log = logging.getLogger("pathsnap.api")


def setup_logging(settings: Settings) -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger("pathsnap")
    root.setLevel(logging.INFO)
    for old in [h for h in root.handlers if getattr(h, "_pathsnap", False)]:  # re-configurable (tests, reloads)
        root.removeHandler(old)
        old.close()
    for handler in (logging.StreamHandler(), logging.FileHandler(settings.log_dir / "api.log")):
        handler.setFormatter(fmt)
        handler._pathsnap = True  # type: ignore[attr-defined]
        root.addHandler(handler)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    setup_logging(settings)
    service = PredictionService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("PathSnap API starting (artifacts=%s, data=%s)", settings.artifact_dir, settings.eval_data_root)
        service.load()
        yield
        log.info("PathSnap API shutting down")

    app = FastAPI(title="PathSnap map-matching API", version="2.0.0", lifespan=lifespan)
    app.state.service = service
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"])

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        start = time.perf_counter()
        client = request.client.host if request.client else "?"
        log.info("REQUEST  %s %s from %s", request.method, request.url.path, client)
        response = await call_next(request)
        log.info("RESPONSE %s %s -> %d (%.2fs)", request.method, request.url.path, response.status_code,
                 time.perf_counter() - start)
        return response

    @app.get("/health", response_model=HealthResponse)
    def health():
        meta = service.bundle.metadata if service.bundle else {}
        trips = service.trip_ids()
        ok = service.bundle is not None and bool(trips)
        detail = None if ok else (service.load_error or "No evaluation trips available (check PATHSNAP_EVAL_DATA_ROOT)")
        return HealthResponse(status="ok" if ok else "degraded", model_loaded=service.bundle is not None,
                              model_key=meta.get("model", {}).get("key"), bundle_version=meta.get("bundle_version"),
                              dataset_label=settings.dataset_label, eval_data_configured=settings.eval_data_root is not None,
                              n_trips_available=len(trips), detail=detail)

    @app.get("/api/evaluation/trips", response_model=TripListResponse)
    def trips():
        if settings.eval_data_root is None or not settings.eval_data_root.is_dir():
            raise HTTPException(503, "Evaluation data not available on the server (PATHSNAP_EVAL_DATA_ROOT)")
        ids = service.trip_ids()
        warning = None if len(ids) == EXPECTED_EVALUATION_TRIPS else \
            f"Expected {EXPECTED_EVALUATION_TRIPS} evaluation trips, found {len(ids)}"
        return TripListResponse(dataset_label=settings.dataset_label, count=len(ids),
                                expected_count=EXPECTED_EVALUATION_TRIPS, trips=[{"trip_id": t} for t in ids],
                                warning=warning)

    @app.post("/api/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        log.info("predict request for trip %s", req.trip_id)
        try:
            return service.predict(req.trip_id)
        except TripNotFound:
            log.warning("unknown trip id %s", req.trip_id)
            raise HTTPException(404, f"Trip '{req.trip_id}' is not available on this server")
        except ServiceUnavailable as exc:
            log.error("service unavailable: %s", exc)
            raise HTTPException(503, str(exc))
        except Exception as exc:  # noqa: BLE001 - report, log and keep serving
            log.exception("prediction FAILED for trip %s", req.trip_id)
            raise HTTPException(500, f"Prediction failed: {type(exc).__name__}: {exc}")

    @app.get("/api/evaluation/results/{trip_id}", response_model=EvaluationResult)
    def results(trip_id: str):
        if trip_id not in service.results:
            raise HTTPException(404, f"No prediction has been run for trip '{trip_id}' yet")
        return service.results[trip_id]

    return app


app = create_app()
