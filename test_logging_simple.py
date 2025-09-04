#!/usr/bin/env python3
"""
Simple test for enhanced logging system functionality.
"""

import asyncio
import time

from app.logger import LoggingContext, generate_correlation_id, health_checker, logger
from app.monitoring import (
    increment_counter,
    metrics_collector,
    monitoring_dashboard,
    set_gauge,
    system_monitor,
)


async def test_basic_functionality():
    """Test basic enhanced logging functionality"""
    print("=== Testing Enhanced Logging System ===")

    # Test structured logging with correlation ID
    correlation_id = generate_correlation_id()
    with LoggingContext(
        correlation_id=correlation_id, agent_id="test_agent", operation="test_operation"
    ):
        logger.info(
            "Testing enhanced structured logging",
            {"test_data": "sample_value", "number": 42},
        )

        # Test performance logging
        logger.log_performance("test_operation", 0.123, {"status": "success"})

        # Test security event logging
        logger.log_security_event(
            "test_security_event",
            {"user_id": "test_user", "action": "login_attempt"},
            "medium",
        )

    print(f"✓ Structured logging with correlation ID: {correlation_id}")

    # Test metrics collection
    increment_counter("test_requests", 1.0, {"endpoint": "/api/test"})
    set_gauge("active_connections", 15.0)

    metrics_summary = metrics_collector.get_metrics_summary()
    print(f"✓ Metrics collected: {metrics_summary['total_metrics']} total")

    # Test health checks
    health_results = await health_checker.run_health_checks()
    print(
        f"✓ Health checks: {health_results['overall_status']} ({health_results['summary']['total_checks']} checks)"
    )

    # Test system monitoring
    system_metrics = system_monitor.get_system_metrics()
    print(
        f"✓ System monitoring: CPU {system_metrics.cpu_percent:.1f}%, Memory {system_metrics.memory_percent:.1f}%"
    )

    # Test dashboard data
    dashboard_data = await monitoring_dashboard.get_dashboard_data()
    print(f"✓ Dashboard status: {dashboard_data.get('status', 'unknown')}")

    # Test log statistics
    stats = logger.get_log_statistics()
    print(
        f"✓ Log statistics: {stats['total_logs']} total logs, {stats['error_rate_percentage']:.2f}% error rate"
    )

    print("\n✅ All basic functionality tests passed!")
    return True


if __name__ == "__main__":
    asyncio.run(test_basic_functionality())
