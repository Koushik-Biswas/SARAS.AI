"""SARAS AI — Smart Assessment and Recognition Automation System: FastAPI app.

Run:   uvicorn app.main:app            (from backend/)
Tests: create_app(Settings(...)) injects settings every module then reads via
       config.get_settings(); use `with TestClient(app)` to run the lifespan.

Importing this module is cheap: the module-level `app` is only built the first
time it is accessed (e.g. by uvicorn)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings, get_settings, set_settings
from .db import SessionLocal, create_schema, get_engine, init_engine
from .errors import _DEFAULT_CODES, ApiError, install_error_handlers
from .ratelimit import RateLimitMiddleware
from .uploadguard import UploadGuardMiddleware

API_PREFIX = "/api/v1"
log = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("app").setLevel(level.upper())


def _error(code: str, message: str, status: int, details=None) -> JSONResponse:  # noqa: ANN001
    return JSONResponse({"error": {"code": code, "message": message, "details": details}}, status_code=status)


def _system_router() -> APIRouter:
    router = APIRouter(tags=["system"])

    @router.get("/health")
    def health() -> dict:
        """Liveness: the process is up (no auth, no database)."""
        return {"status": "ok"}

    @router.get("/ready")
    def ready(request: Request) -> dict:
        """Readiness: the database answers and the worker thread (if enabled) is alive."""
        s = get_settings()
        try:
            with SessionLocal() as db:
                db.execute(text("SELECT 1"))
            database = "ok"
        except Exception:
            log.exception("Readiness check: database unavailable")
            database = "unavailable"
        worker = getattr(request.app.state, "worker", None)
        if not s.worker_enabled:
            worker_state = "disabled"
        else:
            worker_state = "running" if worker is not None and worker.is_alive() else "stopped"
        if database != "ok" or worker_state == "stopped":
            raise ApiError(503, "service_unavailable", "The service is not ready",
                           {"database": database, "worker": worker_state})
        return {"status": "ready", "database": database, "worker": worker_state}

    return router


def _install_spa(app: FastAPI, dist: Path) -> None:
    """Serve the built React SPA: /assets as static files, any other non-API GET
    that matches no route falls back to index.html (client-side routing)."""
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    def spa_file(path: str) -> Path | None:
        index = dist / "index.html"
        if not index.is_file():
            return None
        candidate = (dist / path.lstrip("/")).resolve()
        try:
            if candidate.is_file() and dist.resolve() in candidate.parents:
                return candidate  # e.g. /favicon.ico, /robots.txt
        except OSError:
            pass
        return index

    @app.get("/", include_in_schema=False)
    def root():  # noqa: ANN202
        index = spa_file("/")
        if index is not None:
            return FileResponse(index)
        return {"name": "SARAS AI API", "docs": "/api/docs", "health": f"{API_PREFIX}/health"}

    @app.exception_handler(StarletteHTTPException)
    async def _http_or_spa(request: Request, exc: StarletteHTTPException):  # noqa: ANN202
        path = request.url.path
        if (exc.status_code == 404 and request.method in ("GET", "HEAD")
                and not path.startswith(("/api/", "/assets/")) and path != "/api"):
            target = spa_file(path)
            if target is not None:
                return FileResponse(target)
        code = _DEFAULT_CODES.get(exc.status_code, "error")
        response = _error(code, str(exc.detail), exc.status_code)
        if exc.headers:
            response.headers.update(exc.headers)
        return response


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API. Passing `settings` installs them as the process-wide settings
    (read by every module through config.get_settings())."""
    if settings is not None:
        set_settings(settings)
    s = get_settings()
    configure_logging(s.log_level)

    # The engine is created up front (cheap, no connection) so the app also works
    # without its lifespan running; the schema is created idempotently.
    init_engine(s.database_url)
    create_schema()

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN202
        from .worker import Worker  # imported lazily: pulls in the pipeline (cv2, fitz)

        worker = None
        if get_settings().worker_enabled:
            worker = Worker()
            worker.start()
        app.state.worker = worker
        log.info("SARAS AI API ready (database=%s, worker=%s)", s.database_url.split(":", 1)[0],
                 "running" if worker else "disabled")
        try:
            yield
        finally:
            if worker is not None:
                worker.stop()
                worker.join(timeout=30)
            app.state.worker = None
            try:
                get_engine().dispose()  # release SQLite file handles (Windows)
            except RuntimeError:
                pass

    app = FastAPI(
        title="SARAS AI API",
        summary="Smart Assessment and Recognition Automation System",
        description="AI-powered exam answer-sheet evaluation and marks assessment. "
                    f"All endpoints live under {API_PREFIX}; send `Authorization: Bearer <access token>` "
                    "except for register, login, refresh and health.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.worker = None
    install_error_handlers(app)

    @app.exception_handler(OperationalError)
    async def _db_unavailable(_: Request, exc: OperationalError):  # noqa: ANN202
        log.error("Database error: %s", exc)
        return _error("service_unavailable", "The database is busy or unavailable; please retry", 503)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):  # noqa: ANN202
        log.exception("Unhandled error", exc_info=exc)
        return _error("internal_error", "Something went wrong on the server", 500)

    # Middleware: the last added runs first, so CORS wraps rate limiting and even
    # 429 responses carry CORS headers. The upload guard runs after rate limiting and
    # before the multipart parser spools a single byte to disk.
    app.add_middleware(UploadGuardMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origins,
        allow_credentials=False,  # bearer tokens, no cookies
        allow_methods=["*"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "Accept"],
        expose_headers=["Content-Disposition", "Retry-After"],
    )

    from .api import auth, exams, submissions  # imported here so `import app.main` stays light

    app.include_router(auth.router, prefix=API_PREFIX)
    app.include_router(exams.router, prefix=API_PREFIX)
    app.include_router(submissions.router, prefix=API_PREFIX)
    system = _system_router()
    app.include_router(system, prefix=API_PREFIX)
    app.include_router(system, include_in_schema=False)  # /health and /ready at the root too

    _install_spa(app, Path(s.frontend_dist))
    return app


_app: FastAPI | None = None


def __getattr__(name: str):  # noqa: ANN202 - PEP 562: build `app` lazily on first access
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
