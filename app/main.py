"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    routes_drone,
    routes_health,
    routes_model,
    routes_monitoring,
    routes_observations,
    routes_risk,
    routes_segmentation,
)
from app.config import settings

app = FastAPI(title=settings.app_name, version=settings.app_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(routes_health.router)
app.include_router(routes_observations.router)
app.include_router(routes_risk.router)
app.include_router(routes_drone.router)
app.include_router(routes_model.router)
app.include_router(routes_segmentation.router)
app.include_router(routes_monitoring.router)
