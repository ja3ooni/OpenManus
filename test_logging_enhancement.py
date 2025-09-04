#!/usr/bin/env python3
"""
Test script for enhanced logging and monitoring system.

This script tests the enhanced logging capabilities including:
- Structured logging with correlation IDs
- Performance metrics collection
- Health check endpoints
- Security event logging
- Alert management
"""

import asyncio
import json
import time
from datetime import datetime

from app.logger import (
    LoggingContext,
    generate_correlation_id,
    health_checker,
    logger,
    set_agent_id,
    set_correlation_id,
    set_operation,
)
from app.monitoring import (
    AlertSeverity,
    increment_counter,
    metrics_collector,
    monitoring_dashboard,
    record_histogram,
    set_gauge,
    system_monitor,
    timer,
)


async def test_structured_logging():
    """Test structured logging with correlation IDs"""
    print("=== Testing Structured Logging ===")

    # Test basic logging
    logger.info("Testing basic structured logging")

    # Test logging with context
    with LoggingContext(
        correlation_id=generate_correlation_id(),
        agent_id="test_agent",
        operation="test_operation",
    ):
        logger.info(
            "Testing logging with context",
            {"test_data": "sample_value", "number": 42, "nested": {"key": "value"}},
        )

        # Test error logging
        try:
            raise ValueError("Test error for logging")
        except Exception as e:
            logger.error(
                "Test error occurred", {"error_type": type(e).__name__}, exc_info=True
            )

    # Test performance logging
    logger.log_performance("test_operation", 0.123, {"status": "success"})

    # Test security event logging
    logger.log_security_event(
        "test_security_event",
        {"user_id": "test_user", "action": "login_attempt"},
        "medium",
    )

    # Test business event logging
    logger.log_business_event(
        "user_action", {"action": "file_upload", "file_size": 1024}
    )

    print("✓ Structured logging tests completed")


async def test_performance_metrics():
    """Test performance metrics collection"""
    print("\n=== Testing Performance Metrics ===")

    # Test counters
    increment_counter("test_requests", 1.0, {"endpoint": "/api/test"})
    increment_counter("test_requests", 2.0, {"endpoint": "/api/test"})

    # Test gauges
    set_gauge("active_connections", 15.0)
    set_gauge("queue_size", 8.0)

    # Test histograms
    record_histogram("request_size", 1024.0, {"type": "upload"})
    record_histogram("request_size", 2048.0, {"type": "upload"})
    record_histogram("request_size", 512.0, {"type": "download"})

    # Test timer decorator
    @timer("test_function_timer")
    async def slow_function():
        await asyncio.sleep(0.1)
        return "completed"

    result = await slow_function()

    # Test manual timer
    start_time = time.time()
    await asyncio.sleep(0.05)
    duration = time.time() - start_time
    metrics_collector.record_timer("manual_timer", duration, {"type": "manual"})

    # Get metrics summary
    summary = metrics_collector.get_metrics_summary()
    print(f"✓ Metrics collected: {summary}")

    # Test specific metric retrieval
    counter_value = metrics_collector.get_counter_value(
        "test_requests", {"endpoint": "/api/test"}
    )
    print(f"✓ Counter value: {counter_value}")

    gauge_value = metrics_collector.get_gauge_value("active_connections")
    print(f"✓ Gauge value: {gauge_value}")

    histogram_stats = metrics_collector.get_histogram_stats(
        "request_size", {"type": "upload"}
    )
    print(f"✓ Histogram stats: {histogram_stats}")

    timer_stats = metrics_collector.get_timer_stats("test_function_timer")
    print(f"✓ Timer stats: {timer_stats}")


async def test_health_checks():
    """Test health check system"""
    print("\n=== Testing Health Checks ===")

    # Add a custom health check
    def custom_health_check():
        return {
            "status": "healthy",
            "message": "Custom check passed",
            "details": {"test_value": 42},
        }

    health_checker.register_check("custom_test", custom_health_check)

    # Run health checks
    health_results = await health_checker.run_health_checks()
    print(f"✓ Health check results: {health_results['overall_status']}")
    print(f"✓ Checks completed: {health_results['summary']['total_checks']}")

    # Test health check history
    history = health_checker.get_check_history(5)
    print(f"✓ Health check history entries: {len(history)}")

    return health_results


async def test_monitoring_dashboard():
    """Test monitoring dashboard data collection"""
    print("\n=== Testing Monitoring Dashboard ===")

    # Get comprehensive dashboard data
    dashboard_data = await monitoring_dashboard.get_dashboard_data()

    print(f"✓ Dashboard status: {dashboard_data.get('status', 'unknown')}")
    print(f"✓ Uptime: {dashboard_data.get('uptime_seconds', 0):.2f} seconds")

    # Check system metrics
    if "system" in dashboard_data:
        system = dashboard_data["system"]
        print(f"✓ CPU usage: {system.get('cpu_percent', 0):.1f}%")
        print(f"✓ Memory usage: {system.get('memory_percent', 0):.1f}%")
        print(f"✓ Disk usage: {system.get('disk_percent', 0):.1f}%")

    # Check alerts
    if "alerts" in dashboard_data:
        alerts = dashboard_data["alerts"]
        print(
            f"✓ Active alerts: {alerts.get('summary', {}).get('active_alerts_count', 0)}"
        )

    # Test Prometheus metrics export
    prometheus_metrics = monitoring_dashboard.get_prometheus_metrics()
    print(
        f"✓ Prometheus metrics generated: {len(prometheus_metrics.split('\\n'))} lines"
    )

    return dashboard_data


async def test_alert_system():
    """Test alert management system"""
    print("\n=== Testing Alert System ===")

    # Add custom alert rules
    monitoring_dashboard.alert_manager.add_alert_rule(
        "test_high_counter",
        "counters.test_requests",
        2.0,
        "greater_than",
        AlertSeverity.MEDIUM,
    )

    # Trigger some metrics that should cause alerts
    increment_counter("test_requests", 5.0)  # This should trigger the alert

    # Simulate dashboard data collection which checks alerts
    dashboard_data = await monitoring_dashboard.get_dashboard_data()

    active_alerts = monitoring_dashboard.alert_manager.get_active_alerts()
    alert_summary = monitoring_dashboard.alert_manager.get_alert_summary()

    print(f"✓ Active alerts: {len(active_alerts)}")
    print(f"✓ Alert rules: {alert_summary.get('total_rules', 0)}")

    if active_alerts:
        for alert in active_alerts:
            print(
                f"  - {alert.name}: {alert.message} (severity: {alert.severity.value})"
            )


async def test_system_monitoring():
    """Test system resource monitoring"""
    print("\n=== Testing System Monitoring ===")

    # Get system metrics
    system_metrics = system_monitor.get_system_metrics()

    print(f"✓ CPU: {system_metrics.cpu_percent:.1f}%")
    print(
        f"✓ Memory: {system_metrics.memory_percent:.1f}% ({system_metrics.memory_used_mb:.1f}MB used)"
    )
    print(
        f"✓ Disk: {system_metrics.disk_percent:.1f}% ({system_metrics.disk_free_gb:.1f}GB free)"
    )
    print(f"✓ Processes: {system_metrics.process_count}")
    print(f"✓ Open files: {system_metrics.open_files}")

    # Get process information
    process_info = system_monitor.get_process_info()
    print(f"✓ Process PID: {process_info.get('pid', 'unknown')}")
    print(f"✓ Process name: {process_info.get('name', 'unknown')}")
    print(f"✓ Process threads: {process_info.get('num_threads', 0)}")


async def test_log_statistics():
    """Test logging statistics and monitoring"""
    print("\n=== Testing Log Statistics ===")

    # Generate some log entries
    logger.debug("Debug message for statistics")
    logger.info("Info message for statistics")
    logger.warning("Warning message for statistics")

    # Get log statistics
    stats = logger.get_log_statistics()

    print(f"✓ Total logs: {stats.get('total_logs', 0)}")
    print(f"✓ Error rate: {stats.get('error_rate_percentage', 0):.2f}%")
    print(f"✓ Log counts: {stats.get('log_counts', {})}")

    if stats.get("recent_alerts"):
        print(f"✓ Recent alerts: {len(stats['recent_alerts'])}")

    if stats.get("performance_stats"):
        perf_stats = stats["performance_stats"]
        print(f"✓ Performance uptime: {perf_stats.get('uptime_seconds', 0):.2f}s")


async def test_correlation_tracking():
    """Test correlation ID tracking across operations"""
    print("\n=== Testing Correlation Tracking ===")

    correlation_id = generate_correlation_id()

    # Test manual context setting
    set_correlation_id(correlation_id)
    set_agent_id("test_agent_123")
    set_operation("correlation_test")

    logger.info("Message with manual correlation context", {"test_correlation": True})

    # Test context manager
    with LoggingContext(
        correlation_id=correlation_id,
        request_id="req_123",
        agent_id="agent_456",
        operation="context_manager_test",
    ):
        logger.info("Message within context manager", {"nested_operation": True})

        # Simulate nested operation
        with LoggingContext(operation="nested_operation"):
            logger.info("Nested operation message")

    print(f"✓ Correlation ID used: {correlation_id}")


async def main():
    """Run all logging and monitoring tests"""
    print("Starting Enhanced Logging and Monitoring System Tests")
    print("=" * 60)

    try:
        # Run all test functions
        await test_structured_logging()
        await test_performance_metrics()
        await test_health_checks()
        await test_monitoring_dashboard()
        await test_alert_system()
        await test_system_monitoring()
        await test_log_statistics()
        await test_correlation_tracking()

        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")

        # Final summary
        print("\n=== Final System Summary ===")
        dashboard_data = await monitoring_dashboard.get_dashboard_data()

        print(f"System Status: {dashboard_data.get('status', 'unknown')}")
        print(
            f"Health Status: {dashboard_data.get('health', {}).get('overall_status', 'unknown')}"
        )
        print(
            f"Active Alerts: {dashboard_data.get('alerts', {}).get('summary', {}).get('active_alerts_count', 0)}"
        )
        print(f"Uptime: {dashboard_data.get('uptime_seconds', 0):.2f} seconds")

        # Show metrics summary
        metrics_summary = metrics_collector.get_metrics_summary()
        print(f"Metrics Collected: {metrics_summary.get('total_metrics', 0)}")
        print(f"Counters: {metrics_summary.get('counters_count', 0)}")
        print(f"Gauges: {metrics_summary.get('gauges_count', 0)}")
        print(f"Timers: {metrics_summary.get('timers_count', 0)}")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.exception("Test execution failed")
        raise


if __name__ == "__main__":
    asyncio.run(main())
