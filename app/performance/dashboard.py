"""
Performance monitoring dashboard for OpenManus.

This module provides comprehensive performance dashboards and alerting
capabilities, integrating with the metrics collection system to provide
real-time performance insights and automated alerting.
"""

import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from app.config import config
from app.logger import logger
from app.monitoring import (
    AlertSeverity,
    MonitoringDashboard,
    alert_manager,
    monitoring_dashboard,
    system_monitor,
)
from app.performance.metrics import (
    PerformanceLevel,
    SLAStatus,
    performance_benchmark_runner,
    performance_metrics_collector,
)
from app.performance.testing import TestStatus, TestType, performance_test_runner


class PerformanceDashboard:
    """Enhanced dashboard focused on performance monitoring and alerting"""

    def __init__(
        self,
        base_dashboard: MonitoringDashboard = None,
        metrics_collector=None,
        benchmark_runner=None,
        test_runner=None,
    ):
        self.base_dashboard = base_dashboard or monitoring_dashboard
        self.metrics_collector = metrics_collector or performance_metrics_collector
        self.benchmark_runner = benchmark_runner or performance_benchmark_runner
        self.test_runner = test_runner or performance_test_runner
        self.dashboard_config = self._load_dashboard_config()

        # Set up performance-specific alert rules
        self._setup_performance_alerts()

    def _load_dashboard_config(self) -> Dict[str, Any]:
        """Load dashboard configuration"""
        return {
            "refresh_interval_seconds": 30,
            "data_retention_hours": 24,
            "alert_thresholds": {
                "response_time_warning_ms": 20000,
                "response_time_critical_ms": 40000,
                "throughput_warning_rps": 0.1,
                "error_rate_warning_percent": 5.0,
                "error_rate_critical_percent": 15.0,
                "sla_warning_percent": 90.0,
                "sla_critical_percent": 80.0,
            },
            "performance_targets": {
                "agent_response_time_ms": 30000,
                "tool_execution_time_ms": 10000,
                "llm_api_response_time_ms": 15000,
                "system_availability_percent": 99.5,
                "error_rate_percent": 5.0,
            },
        }

    def _setup_performance_alerts(self):
        """Set up performance-specific alert rules"""
        thresholds = self.dashboard_config["alert_thresholds"]

        # Response time alerts
        alert_manager.add_alert_rule(
            "performance_response_time_warning",
            "performance.avg_response_time_ms",
            thresholds["response_time_warning_ms"],
            "greater_than",
            AlertSeverity.MEDIUM,
            {"category": "performance", "type": "response_time"},
        )

        alert_manager.add_alert_rule(
            "performance_response_time_critical",
            "performance.avg_response_time_ms",
            thresholds["response_time_critical_ms"],
            "greater_than",
            AlertSeverity.CRITICAL,
            {"category": "performance", "type": "response_time"},
        )

        # Throughput alerts
        alert_manager.add_alert_rule(
            "performance_low_throughput",
            "performance.throughput_rps",
            thresholds["throughput_warning_rps"],
            "less_than",
            AlertSeverity.MEDIUM,
            {"category": "performance", "type": "throughput"},
        )

        # Error rate alerts
        alert_manager.add_alert_rule(
            "performance_error_rate_warning",
            "performance.error_rate_percent",
            thresholds["error_rate_warning_percent"],
            "greater_than",
            AlertSeverity.MEDIUM,
            {"category": "performance", "type": "error_rate"},
        )

        alert_manager.add_alert_rule(
            "performance_error_rate_critical",
            "performance.error_rate_percent",
            thresholds["error_rate_critical_percent"],
            "greater_than",
            AlertSeverity.CRITICAL,
            {"category": "performance", "type": "error_rate"},
        )

        # SLA compliance alerts
        alert_manager.add_alert_rule(
            "sla_compliance_warning",
            "performance.sla_compliance_percent",
            thresholds["sla_warning_percent"],
            "less_than",
            AlertSeverity.MEDIUM,
            {"category": "performance", "type": "sla"},
        )

        alert_manager.add_alert_rule(
            "sla_compliance_critical",
            "performance.sla_compliance_percent",
            thresholds["sla_critical_percent"],
            "less_than",
            AlertSeverity.CRITICAL,
            {"category": "performance", "type": "sla"},
        )

    async def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        try:
            # Get base dashboard data
            base_data = await self.base_dashboard.get_dashboard_data()

            # Get performance-specific data
            performance_summary = self.metrics_collector.get_performance_summary()
            sla_reports = self.metrics_collector.get_all_sla_reports()
            benchmark_trends = self.benchmark_runner.get_performance_trends()

            # Get recent benchmark results
            recent_benchmarks = {}
            for benchmark_name in self.benchmark_runner.benchmarks.keys():
                recent_results = self.benchmark_runner.get_benchmark_history(
                    benchmark_name, 5
                )
                recent_benchmarks[benchmark_name] = [
                    asdict(result) for result in recent_results
                ]

            # Get performance test results
            test_summary = self.test_runner.get_test_summary()
            recent_test_results = {}
            for test_name in self.test_runner.test_configs.keys():
                recent_results = self.test_runner.get_test_history(test_name, 3)
                recent_test_results[test_name] = [
                    asdict(result) for result in recent_results
                ]

            # Calculate performance scores
            performance_scores = self._calculate_performance_scores()

            # Get performance alerts
            performance_alerts = self._get_performance_alerts()

            # Determine overall performance status
            overall_performance_status = self._determine_overall_performance_status(
                performance_summary, sla_reports, performance_alerts
            )

            # Create enhanced dashboard data
            dashboard_data = {
                **base_data,
                "performance": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "overall_status": overall_performance_status,
                    "summary": performance_summary,
                    "sla_reports": {
                        name: asdict(report) for name, report in sla_reports.items()
                    },
                    "benchmark_trends": benchmark_trends,
                    "recent_benchmarks": recent_benchmarks,
                    "performance_scores": performance_scores,
                    "alerts": performance_alerts,
                    "targets": self.dashboard_config["performance_targets"],
                    "thresholds": self.dashboard_config["alert_thresholds"],
                    "automated_testing": {
                        "test_summary": test_summary,
                        "recent_test_results": recent_test_results,
                        "test_configs": {
                            name: asdict(config)
                            for name, config in self.test_runner.test_configs.items()
                        },
                    },
                },
            }

            # Update performance metrics for alerting
            self._update_performance_metrics_for_alerting(
                performance_summary, sla_reports
            )

            return dashboard_data

        except Exception as e:
            logger.error(f"Failed to get performance dashboard data: {e}")
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "error": str(e),
                "performance": {
                    "overall_status": "unknown",
                    "error": str(e),
                },
            }

    def _calculate_performance_scores(self) -> Dict[str, Any]:
        """Calculate performance scores for different metrics"""
        scores = {
            "response_time_score": 0,
            "throughput_score": 0,
            "error_rate_score": 0,
            "sla_compliance_score": 0,
            "overall_score": 0,
        }

        try:
            # Get recent performance data
            targets = self.dashboard_config["performance_targets"]

            # Response time score (based on agent response times)
            agent_stats = self.metrics_collector.get_response_time_stats("agent", 60)
            if agent_stats and "avg_ms" in agent_stats:
                avg_response_time = agent_stats["avg_ms"]
                target_response_time = targets["agent_response_time_ms"]
                scores["response_time_score"] = max(
                    0, 100 - (avg_response_time / target_response_time) * 100
                )

            # Throughput score (based on recent throughput)
            throughput_stats = self.metrics_collector.get_throughput_stats("agent", 60)
            if throughput_stats and "requests_per_second" in throughput_stats:
                current_rps = throughput_stats["requests_per_second"]
                # Score based on activity level (higher is better up to a point)
                scores["throughput_score"] = min(100, current_rps * 50)

            # Error rate score
            error_stats = self.metrics_collector.get_error_rate_stats("agent", 60)
            if error_stats and "avg_error_rate" in error_stats:
                avg_error_rate = error_stats["avg_error_rate"]
                target_error_rate = targets["error_rate_percent"]
                scores["error_rate_score"] = max(
                    0, 100 - (avg_error_rate / target_error_rate) * 100
                )

            # SLA compliance score
            sla_reports = self.metrics_collector.get_all_sla_reports()
            if sla_reports:
                compliance_scores = [
                    report.compliance_percent for report in sla_reports.values()
                ]
                scores["sla_compliance_score"] = sum(compliance_scores) / len(
                    compliance_scores
                )

            # Overall score (weighted average)
            weights = {
                "response_time_score": 0.3,
                "throughput_score": 0.2,
                "error_rate_score": 0.3,
                "sla_compliance_score": 0.2,
            }

            scores["overall_score"] = sum(
                scores[metric] * weight for metric, weight in weights.items()
            )

        except Exception as e:
            logger.error(f"Failed to calculate performance scores: {e}")

        return scores

    def _get_performance_alerts(self) -> Dict[str, Any]:
        """Get performance-specific alerts"""
        all_alerts = alert_manager.get_active_alerts()
        performance_alerts = [
            alert
            for alert in all_alerts
            if alert.labels.get("category") == "performance"
        ]

        # Group by type
        alerts_by_type = {}
        for alert in performance_alerts:
            alert_type = alert.labels.get("type", "unknown")
            if alert_type not in alerts_by_type:
                alerts_by_type[alert_type] = []
            alerts_by_type[alert_type].append(asdict(alert))

        return {
            "active_count": len(performance_alerts),
            "by_type": alerts_by_type,
            "by_severity": {
                severity.value: len(
                    [a for a in performance_alerts if a.severity == severity]
                )
                for severity in AlertSeverity
            },
        }

    def _determine_overall_performance_status(
        self,
        performance_summary: Dict[str, Any],
        sla_reports: Dict[str, Any],
        performance_alerts: Dict[str, Any],
    ) -> str:
        """Determine overall performance status"""
        try:
            # Check for critical alerts
            critical_alerts = performance_alerts.get("by_severity", {}).get(
                "critical", 0
            )
            if critical_alerts > 0:
                return "critical"

            # Check SLA breaches
            breached_slas = [
                report
                for report in sla_reports.values()
                if report.status == SLAStatus.BREACHED
            ]
            if breached_slas:
                return "degraded"

            # Check for high alerts
            high_alerts = performance_alerts.get("by_severity", {}).get("high", 0)
            if high_alerts > 0:
                return "degraded"

            # Check SLA at-risk status
            at_risk_slas = [
                report
                for report in sla_reports.values()
                if report.status == SLAStatus.AT_RISK
            ]
            if at_risk_slas:
                return "warning"

            # Check medium alerts
            medium_alerts = performance_alerts.get("by_severity", {}).get("medium", 0)
            if medium_alerts > 0:
                return "warning"

            # Check overall performance from summary
            overall_performance = performance_summary.get("overall_performance", "good")
            if overall_performance in ["critical", "poor"]:
                return "critical"
            elif overall_performance == "acceptable":
                return "warning"
            else:
                return "healthy"

        except Exception as e:
            logger.error(f"Failed to determine performance status: {e}")
            return "unknown"

    def _update_performance_metrics_for_alerting(
        self, performance_summary: Dict[str, Any], sla_reports: Dict[str, Any]
    ):
        """Update performance metrics for alert rule evaluation"""
        try:
            # Update response time metrics
            response_times = performance_summary.get("response_times", {})
            if "agent" in response_times and "avg_ms" in response_times["agent"]:
                self.base_dashboard.metrics_collector.set_gauge(
                    "performance.avg_response_time_ms",
                    response_times["agent"]["avg_ms"],
                )

            # Update throughput metrics
            throughput = performance_summary.get("throughput", {})
            if "agent" in throughput and "requests_per_second" in throughput["agent"]:
                self.base_dashboard.metrics_collector.set_gauge(
                    "performance.throughput_rps",
                    throughput["agent"]["requests_per_second"],
                )

            # Update error rate metrics
            error_rates = performance_summary.get("error_rates", {})
            if "agent" in error_rates and "avg_error_rate" in error_rates["agent"]:
                self.base_dashboard.metrics_collector.set_gauge(
                    "performance.error_rate_percent",
                    error_rates["agent"]["avg_error_rate"],
                )

            # Update SLA compliance metrics
            if sla_reports:
                compliance_scores = [
                    report.compliance_percent for report in sla_reports.values()
                ]
                avg_compliance = sum(compliance_scores) / len(compliance_scores)
                self.base_dashboard.metrics_collector.set_gauge(
                    "performance.sla_compliance_percent", avg_compliance
                )

        except Exception as e:
            logger.error(f"Failed to update performance metrics for alerting: {e}")

    async def get_performance_report(
        self, time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=time_range_hours)

            # Get performance summary
            performance_summary = self.metrics_collector.get_performance_summary()

            # Get SLA reports
            sla_reports = self.metrics_collector.get_all_sla_reports()

            # Get benchmark trends
            benchmark_trends = self.benchmark_runner.get_performance_trends()

            # Get recent benchmark results for all benchmarks
            all_benchmark_results = {}
            for benchmark_name in self.benchmark_runner.benchmarks.keys():
                results = self.benchmark_runner.get_benchmark_history(
                    benchmark_name, 10
                )
                all_benchmark_results[benchmark_name] = [
                    asdict(result) for result in results
                ]

            # Calculate performance insights
            insights = self._generate_performance_insights(
                performance_summary, sla_reports, benchmark_trends
            )

            # Generate recommendations
            recommendations = self._generate_performance_recommendations(
                performance_summary, sla_reports, benchmark_trends
            )

            report = {
                "report_metadata": {
                    "generated_at": end_time.isoformat(),
                    "time_range_hours": time_range_hours,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                },
                "executive_summary": {
                    "overall_performance": performance_summary.get(
                        "overall_performance", "unknown"
                    ),
                    "sla_compliance_summary": self._summarize_sla_compliance(
                        sla_reports
                    ),
                    "key_metrics": self._extract_key_metrics(performance_summary),
                    "critical_issues": self._identify_critical_issues(
                        sla_reports, benchmark_trends
                    ),
                },
                "detailed_metrics": {
                    "performance_summary": performance_summary,
                    "sla_reports": {
                        name: asdict(report) for name, report in sla_reports.items()
                    },
                    "benchmark_trends": benchmark_trends,
                    "benchmark_results": all_benchmark_results,
                },
                "insights": insights,
                "recommendations": recommendations,
                "appendix": {
                    "configuration": self.dashboard_config,
                    "alert_rules": self._get_alert_rules_summary(),
                },
            }

            return report

        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {
                "report_metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(e),
                },
                "error": str(e),
            }

    def _generate_performance_insights(
        self,
        performance_summary: Dict[str, Any],
        sla_reports: Dict[str, Any],
        benchmark_trends: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate performance insights from data"""
        insights = []

        try:
            # Response time insights
            response_times = performance_summary.get("response_times", {})
            for operation, stats in response_times.items():
                if "avg_ms" in stats and stats["avg_ms"] > 0:
                    if stats["avg_ms"] > 20000:  # > 20 seconds
                        insights.append(
                            {
                                "type": "performance_concern",
                                "category": "response_time",
                                "operation": operation,
                                "message": f"High average response time for {operation}: {stats['avg_ms']:.0f}ms",
                                "severity": (
                                    "high" if stats["avg_ms"] > 30000 else "medium"
                                ),
                                "data": stats,
                            }
                        )

            # SLA insights
            for sla_name, report in sla_reports.items():
                if report.status == SLAStatus.BREACHED:
                    insights.append(
                        {
                            "type": "sla_breach",
                            "category": "sla_compliance",
                            "sla_name": sla_name,
                            "message": f"SLA '{sla_name}' is breached with {report.compliance_percent:.1f}% compliance",
                            "severity": "critical",
                            "data": asdict(report),
                        }
                    )
                elif report.status == SLAStatus.AT_RISK:
                    insights.append(
                        {
                            "type": "sla_at_risk",
                            "category": "sla_compliance",
                            "sla_name": sla_name,
                            "message": f"SLA '{sla_name}' is at risk with {report.compliance_percent:.1f}% compliance",
                            "severity": "medium",
                            "data": asdict(report),
                        }
                    )

            # Trend insights
            for benchmark_name, trend_data in benchmark_trends.items():
                if trend_data.get("response_time_trend") == "degrading":
                    insights.append(
                        {
                            "type": "performance_degradation",
                            "category": "trends",
                            "benchmark": benchmark_name,
                            "message": f"Response time trend is degrading for {benchmark_name}",
                            "severity": "medium",
                            "data": trend_data,
                        }
                    )

                if trend_data.get("error_rate_trend") == "degrading":
                    insights.append(
                        {
                            "type": "error_rate_increase",
                            "category": "trends",
                            "benchmark": benchmark_name,
                            "message": f"Error rate trend is increasing for {benchmark_name}",
                            "severity": "high",
                            "data": trend_data,
                        }
                    )

        except Exception as e:
            logger.error(f"Failed to generate performance insights: {e}")

        return insights

    def _generate_performance_recommendations(
        self,
        performance_summary: Dict[str, Any],
        sla_reports: Dict[str, Any],
        benchmark_trends: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate performance improvement recommendations"""
        recommendations = []

        try:
            # Response time recommendations
            response_times = performance_summary.get("response_times", {})
            for operation, stats in response_times.items():
                if "avg_ms" in stats and stats["avg_ms"] > 15000:
                    recommendations.append(
                        {
                            "category": "response_time",
                            "priority": "high" if stats["avg_ms"] > 30000 else "medium",
                            "title": f"Optimize {operation} response time",
                            "description": f"Average response time of {stats['avg_ms']:.0f}ms exceeds target",
                            "suggested_actions": [
                                "Review and optimize LLM API calls",
                                "Implement response caching where appropriate",
                                "Consider request timeout optimization",
                                "Analyze tool execution performance",
                            ],
                            "data": stats,
                        }
                    )

            # SLA recommendations
            breached_slas = [
                (name, report)
                for name, report in sla_reports.items()
                if report.status in [SLAStatus.BREACHED, SLAStatus.AT_RISK]
            ]

            if breached_slas:
                recommendations.append(
                    {
                        "category": "sla_compliance",
                        "priority": "critical",
                        "title": "Address SLA compliance issues",
                        "description": f"{len(breached_slas)} SLA(s) are not meeting targets",
                        "suggested_actions": [
                            "Review SLA target definitions and adjust if necessary",
                            "Implement performance optimizations for affected operations",
                            "Set up proactive monitoring and alerting",
                            "Consider scaling resources if needed",
                        ],
                        "data": {
                            name: asdict(report) for name, report in breached_slas
                        },
                    }
                )

            # Trend-based recommendations
            degrading_trends = [
                (name, trend)
                for name, trend in benchmark_trends.items()
                if trend.get("response_time_trend") == "degrading"
                or trend.get("error_rate_trend") == "degrading"
            ]

            if degrading_trends:
                recommendations.append(
                    {
                        "category": "performance_trends",
                        "priority": "medium",
                        "title": "Address performance degradation trends",
                        "description": f"{len(degrading_trends)} benchmark(s) showing degrading performance",
                        "suggested_actions": [
                            "Investigate recent changes that might impact performance",
                            "Review system resource utilization",
                            "Consider implementing performance regression testing",
                            "Monitor for external dependency issues",
                        ],
                        "data": {name: trend for name, trend in degrading_trends},
                    }
                )

            # General recommendations
            if not recommendations:
                recommendations.append(
                    {
                        "category": "general",
                        "priority": "low",
                        "title": "Maintain current performance levels",
                        "description": "Performance metrics are within acceptable ranges",
                        "suggested_actions": [
                            "Continue regular performance monitoring",
                            "Consider implementing additional benchmarks",
                            "Review and update SLA targets periodically",
                            "Plan for capacity scaling as usage grows",
                        ],
                    }
                )

        except Exception as e:
            logger.error(f"Failed to generate performance recommendations: {e}")

        return recommendations

    def _summarize_sla_compliance(self, sla_reports: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize SLA compliance status"""
        if not sla_reports:
            return {"total_slas": 0, "meeting": 0, "at_risk": 0, "breached": 0}

        meeting = sum(1 for r in sla_reports.values() if r.status == SLAStatus.MEETING)
        at_risk = sum(1 for r in sla_reports.values() if r.status == SLAStatus.AT_RISK)
        breached = sum(
            1 for r in sla_reports.values() if r.status == SLAStatus.BREACHED
        )

        return {
            "total_slas": len(sla_reports),
            "meeting": meeting,
            "at_risk": at_risk,
            "breached": breached,
            "overall_compliance_percent": (meeting / len(sla_reports)) * 100,
        }

    def _extract_key_metrics(
        self, performance_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract key performance metrics for executive summary"""
        key_metrics = {}

        try:
            # Response times
            response_times = performance_summary.get("response_times", {})
            if "agent" in response_times:
                key_metrics["avg_agent_response_time_ms"] = response_times["agent"].get(
                    "avg_ms", 0
                )

            # Throughput
            throughput = performance_summary.get("throughput", {})
            if "agent" in throughput:
                key_metrics["requests_per_second"] = throughput["agent"].get(
                    "requests_per_second", 0
                )

            # Error rates
            error_rates = performance_summary.get("error_rates", {})
            if "agent" in error_rates:
                key_metrics["error_rate_percent"] = error_rates["agent"].get(
                    "avg_error_rate", 0
                )

        except Exception as e:
            logger.error(f"Failed to extract key metrics: {e}")

        return key_metrics

    def _identify_critical_issues(
        self, sla_reports: Dict[str, Any], benchmark_trends: Dict[str, Any]
    ) -> List[str]:
        """Identify critical performance issues"""
        issues = []

        try:
            # SLA breaches
            breached_slas = [
                name
                for name, report in sla_reports.items()
                if report.status == SLAStatus.BREACHED
            ]
            if breached_slas:
                issues.append(f"SLA breaches: {', '.join(breached_slas)}")

            # Degrading trends
            degrading_benchmarks = [
                name
                for name, trend in benchmark_trends.items()
                if trend.get("response_time_trend") == "degrading"
            ]
            if degrading_benchmarks:
                issues.append(
                    f"Degrading performance: {', '.join(degrading_benchmarks)}"
                )

            # High error rates
            high_error_benchmarks = [
                name
                for name, trend in benchmark_trends.items()
                if trend.get("error_rate_trend") == "degrading"
            ]
            if high_error_benchmarks:
                issues.append(
                    f"Increasing error rates: {', '.join(high_error_benchmarks)}"
                )

        except Exception as e:
            logger.error(f"Failed to identify critical issues: {e}")

        return issues

    def _get_alert_rules_summary(self) -> Dict[str, Any]:
        """Get summary of configured alert rules"""
        try:
            return {
                "total_rules": len(alert_manager.alert_rules),
                "enabled_rules": sum(
                    1
                    for rule in alert_manager.alert_rules.values()
                    if rule.get("enabled", True)
                ),
                "performance_rules": sum(
                    1
                    for rule in alert_manager.alert_rules.values()
                    if rule.get("labels", {}).get("category") == "performance"
                ),
            }
        except Exception as e:
            logger.error(f"Failed to get alert rules summary: {e}")
            return {"error": str(e)}

    async def run_performance_health_check(self) -> Dict[str, Any]:
        """Run comprehensive performance health check"""
        try:
            health_results = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "overall_status": "healthy",
                "checks": {},
            }

            # Check response times
            response_check = await self._check_response_times()
            health_results["checks"]["response_times"] = response_check

            # Check SLA compliance
            sla_check = await self._check_sla_compliance()
            health_results["checks"]["sla_compliance"] = sla_check

            # Check error rates
            error_check = await self._check_error_rates()
            health_results["checks"]["error_rates"] = error_check

            # Check system resources
            resource_check = await self._check_system_resources()
            health_results["checks"]["system_resources"] = resource_check

            # Determine overall status
            failed_checks = [
                name
                for name, check in health_results["checks"].items()
                if check["status"] != "healthy"
            ]

            if failed_checks:
                if any(
                    health_results["checks"][name]["status"] == "critical"
                    for name in failed_checks
                ):
                    health_results["overall_status"] = "critical"
                else:
                    health_results["overall_status"] = "degraded"

            health_results["failed_checks"] = failed_checks
            health_results["total_checks"] = len(health_results["checks"])

            return health_results

        except Exception as e:
            logger.error(f"Performance health check failed: {e}")
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "overall_status": "error",
                "error": str(e),
            }

    async def _check_response_times(self) -> Dict[str, Any]:
        """Check response time health"""
        try:
            agent_stats = self.metrics_collector.get_response_time_stats("agent", 30)
            if not agent_stats or "avg_ms" not in agent_stats:
                return {
                    "status": "healthy",
                    "message": "No recent response time data",
                    "details": {},
                }

            avg_response_time = agent_stats["avg_ms"]
            target = self.dashboard_config["performance_targets"][
                "agent_response_time_ms"
            ]

            if avg_response_time > target * 1.5:  # 150% of target
                status = "critical"
                message = f"Response time critically high: {avg_response_time:.0f}ms"
            elif avg_response_time > target:
                status = "degraded"
                message = f"Response time above target: {avg_response_time:.0f}ms"
            else:
                status = "healthy"
                message = f"Response time within target: {avg_response_time:.0f}ms"

            return {
                "status": status,
                "message": message,
                "details": {
                    "current_avg_ms": avg_response_time,
                    "target_ms": target,
                    "stats": agent_stats,
                },
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to check response times: {e}",
                "details": {"error": str(e)},
            }

    async def _check_sla_compliance(self) -> Dict[str, Any]:
        """Check SLA compliance health"""
        try:
            sla_reports = self.metrics_collector.get_all_sla_reports()
            if not sla_reports:
                return {
                    "status": "healthy",
                    "message": "No SLA targets configured",
                    "details": {},
                }

            breached = [
                r for r in sla_reports.values() if r.status == SLAStatus.BREACHED
            ]
            at_risk = [r for r in sla_reports.values() if r.status == SLAStatus.AT_RISK]

            if breached:
                status = "critical"
                message = f"{len(breached)} SLA(s) breached"
            elif at_risk:
                status = "degraded"
                message = f"{len(at_risk)} SLA(s) at risk"
            else:
                status = "healthy"
                message = "All SLAs meeting targets"

            return {
                "status": status,
                "message": message,
                "details": {
                    "total_slas": len(sla_reports),
                    "breached": len(breached),
                    "at_risk": len(at_risk),
                    "meeting": len(sla_reports) - len(breached) - len(at_risk),
                },
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to check SLA compliance: {e}",
                "details": {"error": str(e)},
            }

    async def _check_error_rates(self) -> Dict[str, Any]:
        """Check error rate health"""
        try:
            error_stats = self.metrics_collector.get_error_rate_stats("agent", 30)
            if not error_stats or "avg_error_rate" not in error_stats:
                return {
                    "status": "healthy",
                    "message": "No recent error rate data",
                    "details": {},
                }

            avg_error_rate = error_stats["avg_error_rate"]
            target = self.dashboard_config["performance_targets"]["error_rate_percent"]

            if avg_error_rate > target * 3:  # 3x target
                status = "critical"
                message = f"Error rate critically high: {avg_error_rate:.1f}%"
            elif avg_error_rate > target:
                status = "degraded"
                message = f"Error rate above target: {avg_error_rate:.1f}%"
            else:
                status = "healthy"
                message = f"Error rate within target: {avg_error_rate:.1f}%"

            return {
                "status": status,
                "message": message,
                "details": {
                    "current_error_rate": avg_error_rate,
                    "target_error_rate": target,
                    "stats": error_stats,
                },
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to check error rates: {e}",
                "details": {"error": str(e)},
            }

    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource health"""
        try:
            system_metrics = system_monitor.get_system_metrics()

            issues = []
            if system_metrics.cpu_percent > 90:
                issues.append(f"High CPU usage: {system_metrics.cpu_percent:.1f}%")
            if system_metrics.memory_percent > 90:
                issues.append(
                    f"High memory usage: {system_metrics.memory_percent:.1f}%"
                )
            if system_metrics.disk_percent > 95:
                issues.append(f"High disk usage: {system_metrics.disk_percent:.1f}%")

            if issues:
                status = (
                    "critical"
                    if any("High" in issue for issue in issues)
                    else "degraded"
                )
                message = "; ".join(issues)
            else:
                status = "healthy"
                message = "System resources within normal ranges"

            return {
                "status": status,
                "message": message,
                "details": asdict(system_metrics),
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to check system resources: {e}",
                "details": {"error": str(e)},
            }

    async def run_automated_performance_tests(
        self,
        test_filter: Optional[List[str]] = None,
        test_function: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run automated performance tests and return results"""
        try:
            logger.info("Starting automated performance tests")

            # Run tests
            test_results = await self.test_runner.run_all_tests(
                test_function=test_function, test_filter=test_filter
            )

            # Analyze results
            analysis = {
                "execution_summary": {
                    "total_tests": len(test_results),
                    "successful_tests": len(
                        [r for r in test_results if r.status == TestStatus.COMPLETED]
                    ),
                    "failed_tests": len(
                        [r for r in test_results if r.status == TestStatus.FAILED]
                    ),
                    "cancelled_tests": len(
                        [r for r in test_results if r.status == TestStatus.CANCELLED]
                    ),
                },
                "performance_analysis": {},
                "recommendations": [],
                "test_results": [asdict(result) for result in test_results],
            }

            # Analyze each test result
            for result in test_results:
                if result.status == TestStatus.COMPLETED:
                    analysis["performance_analysis"][result.test_name] = {
                        "performance_level": result.performance_level.value,
                        "meets_targets": result.meets_targets,
                        "avg_response_time_ms": result.avg_response_time_ms,
                        "requests_per_second": result.requests_per_second,
                        "error_rate_percent": result.error_rate_percent,
                        "test_type": result.test_type.value,
                    }

                    # Generate recommendations based on results
                    if not result.meets_targets:
                        analysis["recommendations"].append(
                            {
                                "test_name": result.test_name,
                                "issue": "Performance targets not met",
                                "details": {
                                    "avg_response_time_ms": result.avg_response_time_ms,
                                    "requests_per_second": result.requests_per_second,
                                    "error_rate_percent": result.error_rate_percent,
                                },
                                "suggested_actions": [
                                    "Review system resource utilization",
                                    "Optimize slow operations",
                                    "Consider scaling resources",
                                    "Investigate error causes",
                                ],
                            }
                        )

            logger.info(
                "Automated performance tests completed",
                {
                    "total_tests": analysis["execution_summary"]["total_tests"],
                    "successful_tests": analysis["execution_summary"][
                        "successful_tests"
                    ],
                    "failed_tests": analysis["execution_summary"]["failed_tests"],
                },
            )

            return analysis

        except Exception as e:
            logger.error(f"Failed to run automated performance tests: {e}")
            return {
                "error": str(e),
                "execution_summary": {
                    "total_tests": 0,
                    "successful_tests": 0,
                    "failed_tests": 0,
                },
                "test_results": [],
            }

    async def run_regression_tests(
        self, test_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run regression tests against baselines"""
        try:
            logger.info("Starting regression tests")

            # Get tests to run
            tests_to_run = test_names or list(self.test_runner.baseline_results.keys())

            if not tests_to_run:
                return {
                    "error": "No baseline results available for regression testing",
                    "regression_results": [],
                }

            regression_results = []

            for test_name in tests_to_run:
                if test_name in self.test_runner.baseline_results:
                    try:
                        regression_result = await self.test_runner.run_regression_test(
                            test_name,
                            regression_threshold_percent=15.0,  # 15% threshold
                        )
                        regression_results.append(regression_result)

                    except Exception as e:
                        logger.error(
                            f"Failed to run regression test for '{test_name}': {e}"
                        )

            # Analyze regression results
            analysis = {
                "regression_summary": {
                    "total_tests": len(regression_results),
                    "regressions_detected": len(
                        [r for r in regression_results if r.regression_detected]
                    ),
                    "performance_improvements": len(
                        [
                            r
                            for r in regression_results
                            if not r.regression_detected
                            and r.performance_change_percent < -5
                        ]
                    ),
                },
                "regression_details": [asdict(result) for result in regression_results],
                "critical_regressions": [
                    asdict(result)
                    for result in regression_results
                    if result.regression_detected
                    and abs(result.performance_change_percent) > 25
                ],
            }

            logger.info(
                "Regression tests completed",
                {
                    "total_tests": analysis["regression_summary"]["total_tests"],
                    "regressions_detected": analysis["regression_summary"][
                        "regressions_detected"
                    ],
                },
            )

            return analysis

        except Exception as e:
            logger.error(f"Failed to run regression tests: {e}")
            return {
                "error": str(e),
                "regression_summary": {"total_tests": 0, "regressions_detected": 0},
                "regression_details": [],
            }


# Global performance dashboard instance
performance_dashboard = PerformanceDashboard()
