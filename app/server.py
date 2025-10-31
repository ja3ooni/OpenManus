"""
HTTP server for OpenManus container health checks and management.

This module provides a lightweight HTTP server for health checks,
metrics, and basic container management endpoints.
"""

import asyncio
import json
import signal
import sys
from datetime import datetime, timezone
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from app.config import config
from app.health import health_checker
from app.logger import logger
from app.monitoring import monitoring_dashboard


class GracefulShutdownHandler:
    """Handles graceful shutdown of the application"""

    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.cleanup_tasks = []
        self._shutdown_in_progress = False

        # Register signal handlers for Unix signals
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._signal_handler)
        if hasattr(signal, "SIGINT"):
            signal.signal(signal.SIGINT, self._signal_handler)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        if self._shutdown_in_progress:
            logger.warning(f"Shutdown already in progress, ignoring signal {signum}")
            return

        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_in_progress = True

        # Create shutdown task in the event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._shutdown())
            else:
                asyncio.run(self._shutdown())
        except RuntimeError:
            # If no event loop is running, create one
            asyncio.run(self._shutdown())

    async def _shutdown(self):
        """Perform graceful shutdown"""
        if self.shutdown_event.is_set():
            return

        logger.info("Starting graceful shutdown...")

        # Run cleanup tasks with timeout
        if self.cleanup_tasks:
            logger.info(f"Running {len(self.cleanup_tasks)} cleanup tasks...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.cleanup_tasks, return_exceptions=True),
                    timeout=30.0,  # 30 second timeout for cleanup
                )
                logger.info("Cleanup tasks completed successfully")
            except asyncio.TimeoutError:
                logger.warning("Cleanup tasks timed out, forcing shutdown")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

        # Set shutdown event
        self.shutdown_event.set()
        logger.info("Graceful shutdown completed")

    def add_cleanup_task(self, task):
        """Add a cleanup task to run during shutdown"""
        if asyncio.iscoroutine(task):
            self.cleanup_tasks.append(task)
        else:
            # Wrap non-coroutine tasks
            async def wrapper():
                if callable(task):
                    return task()
                return task

            self.cleanup_tasks.append(wrapper())

    async def wait_for_shutdown(self):
        """Wait for shutdown signal"""
        await self.shutdown_event.wait()


# Create FastAPI app
app = FastAPI(
    title="OpenManus Container API",
    description="Health checks and management endpoints for OpenManus container",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global shutdown handler
shutdown_handler = GracefulShutdownHandler()


@app.get("/health", response_class=JSONResponse)
async def health_check():
    """
    Comprehensive health check endpoint for load balancers and monitoring.
    Returns detailed health information about all system components.
    """
    try:
        # Run all health checks
        health_results = await health_checker.run_all_checks()
        overall_health = health_checker.get_overall_health()

        # Determine HTTP status code based on health
        if overall_health.status.value in ["critical", "unhealthy"]:
            status_code = 503  # Service Unavailable
        elif overall_health.status.value == "degraded":
            status_code = 200  # OK but with warnings
        else:
            status_code = 200  # OK

        response_data = {
            "status": overall_health.status.value,
            "message": overall_health.message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "duration_ms": result.duration_ms,
                }
                for name, result in health_results.items()
            },
        }

        return JSONResponse(content=response_data, status_code=status_code)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "critical",
                "message": f"Health check system failure: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            status_code=503,
        )


@app.get("/health/live", response_class=PlainTextResponse)
async def liveness_probe():
    """
    Simple liveness probe for Kubernetes/Docker health checks.
    Returns 200 OK if the application is running.
    """
    return "OK"


@app.get("/health/ready", response_class=JSONResponse)
async def readiness_probe():
    """
    Readiness probe to check if the application is ready to serve traffic.
    Checks critical components only.
    """
    try:
        # Check only critical components for readiness
        critical_checks = ["configuration", "storage"]

        results = {}
        all_ready = True

        for check_name in critical_checks:
            result = await health_checker.run_health_check(check_name)
            results[check_name] = {
                "status": result.status.value,
                "message": result.message,
            }

            if result.status.value in ["critical", "unhealthy"]:
                all_ready = False

        status_code = 200 if all_ready else 503

        return JSONResponse(
            content={
                "ready": all_ready,
                "checks": results,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            status_code=status_code,
        )

    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            content={
                "ready": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            status_code=503,
        )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics_endpoint():
    """
    Prometheus-compatible metrics endpoint.
    Returns metrics in Prometheus text format.
    """
    try:
        metrics_text = monitoring_dashboard.get_prometheus_metrics()
        return PlainTextResponse(content=metrics_text)

    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")


@app.get("/status", response_class=JSONResponse)
async def status_endpoint():
    """
    Detailed status endpoint with system information.
    """
    try:
        # Get comprehensive status
        dashboard_data = await monitoring_dashboard.get_dashboard_data()
        health_summary = health_checker.get_health_summary()

        status_data = {
            "application": {
                "name": "OpenManus",
                "version": "1.0.0",
                "environment": getattr(config, "environment", "unknown"),
                "uptime_seconds": health_summary.get("uptime_seconds", 0),
            },
            "health": health_summary,
            "system": dashboard_data.get("system", {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return JSONResponse(content=status_data)

    except Exception as e:
        logger.error(f"Status endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Status unavailable")


@app.get("/info", response_class=JSONResponse)
async def info_endpoint():
    """
    Basic application information endpoint.
    """
    return {
        "name": "OpenManus",
        "description": "Production-ready AI agent framework",
        "version": "1.0.0",
        "environment": getattr(config, "environment", "unknown"),
        "build_date": datetime.now(timezone.utc).isoformat(),
        "endpoints": {
            "health": "/health",
            "liveness": "/health/live",
            "readiness": "/health/ready",
            "metrics": "/metrics",
            "status": "/status",
            "docs": "/docs",
        },
    }


@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("OpenManus container server starting up...")

    # Run initial health checks
    try:
        await health_checker.run_all_checks()
        logger.info("Initial health checks completed")
    except Exception as e:
        logger.error(f"Initial health checks failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("OpenManus container server shutting down...")
    await shutdown_handler._shutdown()


async def run_server():
    """Run the container server"""
    # Get configuration
    host = "0.0.0.0"
    port = 8080

    # Configure uvicorn
    config_uvicorn = uvicorn.Config(
        app, host=host, port=port, log_level="info", access_log=True, loop="asyncio"
    )

    server = uvicorn.Server(config_uvicorn)

    # Add server shutdown to cleanup tasks
    shutdown_handler.add_cleanup_task(server.shutdown())

    logger.info(f"Starting OpenManus container server on {host}:{port}")

    # Start server
    await server.serve()


async def main():
    """Main entry point for container server"""
    try:
        # Start the server
        await run_server()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")

    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

    finally:
        # Wait for graceful shutdown
        await shutdown_handler.wait_for_shutdown()


if __name__ == "__main__":
    asyncio.run(main())
