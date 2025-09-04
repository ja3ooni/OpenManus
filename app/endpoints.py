"""
HTTP endpoints for monitoring and health checks.

This module provides HTTP endpoints for system monitoring, health checks,
and metrics collection that can be used by load balancers, monitoring
systems, and dashboards.
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.responses import JSONResponse, PlainTextResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from app.logger import health_checker, logger
from app.monitoring import metrics_collector, monitoring_dashboard, system_monitor


class HealthEndpoints:
    """Health check and monitoring endpoints"""

    def __init__(self):
        self.app = None
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(title="OpenManus Monitoring", version="1.0.0")
            self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes"""
        if not self.app:
            return

        @self.app.get("/health")
        async def health_check():
            """Basic health check endpoint"""
            try:
                health_results = await health_checker.run_health_checks()
                status_code = 200

                if health_results["overall_status"] == "critical":
                    status_code = 503
                elif health_results["overall_status"] in ["degraded", "unhealthy"]:
                    status_code = 503

                return JSONResponse(content=health_results, status_code=status_code)
            except Exception as e:
                logger.error(f"Health check endpoint failed: {e}")
                return JSONResponse(
                    content={
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    status_code=500,
                )

        @self.app.get("/health/live")
        async def liveness_check():
            """Kubernetes liveness probe endpoint"""
            return JSONResponse(
                content={
                    "status": "alive",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                status_code=200,
            )

        @self.app.get("/health/ready")
        async def readiness_check():
            """Kubernetes readiness probe endpoint"""
            try:
                health_results = await health_checker.run_health_checks()

                # Check if critical systems are healthy
                critical_healthy = all(
                    check.get("status") == "healthy"
                    for check in health_results["checks"].values()
                    if check.get("critical", False)
                )

                if critical_healthy:
                    return JSONResponse(
                        content={
                            "status": "ready",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                        status_code=200,
                    )
                else:
                    return JSONResponse(
                        content={
                            "status": "not_ready",
                            "reason": "Critical systems unhealthy",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                        status_code=503,
                    )
            except Exception as e:
                return JSONResponse(
                    content={
                        "status": "not_ready",
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    status_code=503,
                )

        @self.app.get("/metrics")
        async def metrics_endpoint():
            """Prometheus metrics endpoint"""
            try:
                prometheus_metrics = monitoring_dashboard.get_prometheus_metrics()
                return PlainTextResponse(
                    content=prometheus_metrics, media_type="text/plain"
                )
            except Exception as e:
                logger.error(f"Metrics endpoint failed: {e}")
                return PlainTextResponse(
                    content=f"# Error generating metrics: {e}", status_code=500
                )

        @self.app.get("/metrics/json")
        async def metrics_json():
            """JSON metrics endpoint"""
            try:
                dashboard_data = await monitoring_dashboard.get_dashboard_data()
                return JSONResponse(content=dashboard_data)
            except Exception as e:
                logger.error(f"JSON metrics endpoint failed: {e}")
                return JSONResponse(
                    content={
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    status_code=500,
                )

        @self.app.get("/status")
        async def status_endpoint():
            """Comprehensive status endpoint"""
            try:
                dashboard_data = await monitoring_dashboard.get_dashboard_data()
                return JSONResponse(content=dashboard_data)
            except Exception as e:
                logger.error(f"Status endpoint failed: {e}")
                return JSONResponse(
                    content={
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    status_code=500,
                )

        @self.app.get("/info")
        async def info_endpoint():
            """Application information endpoint"""
            try:
                from app.config import config, get_config

                return JSONResponse(
                    content={
                        "application": "OpenManus",
                        "version": "1.0.0",
                        "environment": "production",
                        "uptime_seconds": system_monitor.get_uptime_seconds(),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            except Exception as e:
                logger.error(f"Info endpoint failed: {e}")
                return JSONResponse(
                    content={
                        "application": "OpenManus",
                        "version": "1.0.0",
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    status_code=500,
                )


class SimpleHealthServer:
    """Simple HTTP server for health checks when FastAPI is not available"""

    def __init__(self, port: int = 8080):
        self.port = port

    async def get_health_response(self) -> Dict[str, Any]:
        """Get health check response"""
        try:
            health_results = await health_checker.run_health_checks()
            return health_results
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def get_metrics_response(self) -> Dict[str, Any]:
        """Get metrics response"""
        try:
            dashboard_data = await monitoring_dashboard.get_dashboard_data()
            return dashboard_data
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


# Global endpoint instance
health_endpoints = HealthEndpoints() if FASTAPI_AVAILABLE else None
simple_health_server = SimpleHealthServer()


def get_health_app():
    """Get FastAPI app for health endpoints"""
    if health_endpoints and health_endpoints.app:
        return health_endpoints.app
    return None


async def get_health_status() -> Dict[str, Any]:
    """Get current health status (can be used without FastAPI)"""
    return await simple_health_server.get_health_response()


async def get_metrics_data() -> Dict[str, Any]:
    """Get current metrics data (can be used without FastAPI)"""
    return await simple_health_server.get_metrics_response()


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def test_endpoints():
        print("Testing health status...")
        health = await get_health_status()
        print(json.dumps(health, indent=2, default=str))

        print("\nTesting metrics data...")
        metrics = await get_metrics_data()
        print(json.dumps(metrics, indent=2, default=str))

    asyncio.run(test_endpoints())
