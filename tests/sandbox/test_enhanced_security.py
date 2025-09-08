"""
Tests for enhanced sandbox security features.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from app.sandbox.core.security import (
    NetworkPolicy,
    ResourceLimits,
    SandboxSecurityManager,
    SecurityMetrics,
    SecurityPolicy,
    SecurityViolation,
    SecurityViolationType,
)
from app.security.models import SecurityLevel


@pytest.fixture
def security_policy():
    """Create a test security policy."""
    return SecurityPolicy(
        resource_limits=ResourceLimits(
            memory_mb=256, cpu_percent=25.0, max_processes=10, max_network_connections=5
        ),
        network_policy=NetworkPolicy.NONE,
        blocked_commands={"rm", "sudo", "kill"},
        allowed_file_extensions={".txt", ".py", ".json"},
        blocked_file_patterns={"/etc/", "/bin/"},
        enable_monitoring=True,
        violation_threshold=3,
        auto_terminate_on_violation=True,
    )


@pytest.fixture
def security_manager(security_policy):
    """Create a security manager instance."""
    return SandboxSecurityManager(security_policy)


@pytest.fixture
def mock_container():
    """Create a mock Docker container."""
    container = MagicMock()
    container.id = "test_container_123"
    container.attrs = {"NetworkSettings": {"Networks": {"bridge": {}}}}
    container.stats.return_value = {
        "cpu_stats": {
            "cpu_usage": {"total_usage": 1000000},
            "system_cpu_usage": 10000000,
        },
        "precpu_stats": {
            "cpu_usage": {"total_usage": 900000},
            "system_cpu_usage": 9000000,
        },
        "memory_stats": {"usage": 100 * 1024 * 1024},  # 100MB
    }
    container.exec_run.return_value = MagicMock(
        exit_code=0, output=b"5\n"  # Mock process count
    )
    return container


class TestSandboxSecurityManager:
    """Test cases for SandboxSecurityManager."""

    @pytest.mark.asyncio
    async def test_apply_security_policy(self, security_manager, mock_container):
        """Test applying security policy to container."""
        with patch.object(
            security_manager, "_apply_resource_limits"
        ) as mock_limits, patch.object(
            security_manager, "_apply_network_policy"
        ) as mock_network, patch.object(
            security_manager, "_apply_filesystem_restrictions"
        ) as mock_fs, patch.object(
            security_manager, "start_monitoring"
        ) as mock_monitor:

            await security_manager.apply_security_policy(mock_container)

            mock_limits.assert_called_once_with(mock_container)
            mock_network.assert_called_once_with(mock_container)
            mock_fs.assert_called_once_with(mock_container)
            mock_monitor.assert_called_once_with(mock_container)

    @pytest.mark.asyncio
    async def test_apply_resource_limits(self, security_manager, mock_container):
        """Test applying resource limits."""
        await security_manager._apply_resource_limits(mock_container)

        # Verify container.update was called with correct limits
        mock_container.update.assert_called_once()
        call_args = mock_container.update.call_args[1]
        assert call_args["mem_limit"] == "256m"
        assert call_args["cpu_quota"] == 25000  # 25% of 100000
        assert call_args["pids_limit"] == 10

    @pytest.mark.asyncio
    async def test_network_policy_none(self, security_manager, mock_container):
        """Test network policy NONE disconnects all networks."""
        mock_network = MagicMock()
        mock_container.client.networks.get.return_value = mock_network

        await security_manager._apply_network_policy(mock_container)

        mock_container.client.networks.get.assert_called_with("bridge")
        mock_network.disconnect.assert_called_once_with(mock_container)

    @pytest.mark.asyncio
    async def test_validate_file_access_allowed(self, security_manager):
        """Test file access validation for allowed files."""
        result = await security_manager.validate_file_access(
            "container_id", "/work/test.py"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_file_access_blocked_extension(self, security_manager):
        """Test file access validation for blocked extensions."""
        result = await security_manager.validate_file_access(
            "container_id", "/work/test.exe"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_file_access_blocked_pattern(self, security_manager):
        """Test file access validation for blocked patterns."""
        result = await security_manager.validate_file_access(
            "container_id", "/etc/passwd"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_command_execution_allowed(self, security_manager):
        """Test command validation for allowed commands."""
        result = await security_manager.validate_command_execution(
            "container_id", "python script.py"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_command_execution_blocked(self, security_manager):
        """Test command validation for blocked commands."""
        result = await security_manager.validate_command_execution(
            "container_id", "rm -rf /"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_start_monitoring(self, security_manager, mock_container):
        """Test starting container monitoring."""
        with patch.object(security_manager, "_monitor_container") as mock_monitor:
            mock_monitor.return_value = AsyncMock()

            await security_manager.start_monitoring(mock_container)

            assert mock_container.id in security_manager.monitored_containers
            assert mock_container.id in security_manager.monitoring_tasks

    @pytest.mark.asyncio
    async def test_stop_monitoring(self, security_manager, mock_container):
        """Test stopping container monitoring."""
        # First start monitoring
        mock_task = AsyncMock()
        security_manager.monitored_containers[mock_container.id] = mock_container
        security_manager.monitoring_tasks[mock_container.id] = mock_task

        await security_manager.stop_monitoring(mock_container.id)

        mock_task.cancel.assert_called_once()
        assert mock_container.id not in security_manager.monitored_containers
        assert mock_container.id not in security_manager.monitoring_tasks

    @pytest.mark.asyncio
    async def test_collect_metrics(self, security_manager, mock_container):
        """Test collecting security metrics."""
        metrics = await security_manager._collect_metrics(mock_container)

        assert isinstance(metrics, SecurityMetrics)
        assert metrics.container_id == mock_container.id
        assert metrics.cpu_usage_percent > 0
        assert metrics.memory_usage_mb > 0

    @pytest.mark.asyncio
    async def test_check_violations_cpu_exceeded(
        self, security_manager, mock_container
    ):
        """Test violation detection for CPU usage."""
        metrics = SecurityMetrics(
            container_id=mock_container.id,
            timestamp=asyncio.get_event_loop().time(),
            cpu_usage_percent=50.0,  # Exceeds 25% limit
            memory_usage_mb=100.0,
            disk_usage_mb=0.0,
            network_connections=2,
            process_count=5,
            open_files=10,
            violations_count=0,
        )

        violations = await security_manager._check_violations(mock_container, metrics)

        assert len(violations) == 1
        assert (
            violations[0].violation_type
            == SecurityViolationType.RESOURCE_LIMIT_EXCEEDED
        )
        assert violations[0].details["resource"] == "cpu"

    @pytest.mark.asyncio
    async def test_check_violations_memory_exceeded(
        self, security_manager, mock_container
    ):
        """Test violation detection for memory usage."""
        metrics = SecurityMetrics(
            container_id=mock_container.id,
            timestamp=asyncio.get_event_loop().time(),
            cpu_usage_percent=10.0,
            memory_usage_mb=300.0,  # Exceeds 256MB limit
            disk_usage_mb=0.0,
            network_connections=2,
            process_count=5,
            open_files=10,
            violations_count=0,
        )

        violations = await security_manager._check_violations(mock_container, metrics)

        assert len(violations) == 1
        assert (
            violations[0].violation_type
            == SecurityViolationType.RESOURCE_LIMIT_EXCEEDED
        )
        assert violations[0].details["resource"] == "memory"

    @pytest.mark.asyncio
    async def test_check_suspicious_processes(self, security_manager, mock_container):
        """Test detection of suspicious processes."""
        # Mock ps output with blocked command
        mock_container.exec_run.return_value = MagicMock(
            exit_code=0,
            output=b"USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\nroot         1  0.0  0.1   4624   796 ?        Ss   10:00   0:00 rm -rf /tmp\n",
        )

        violations = await security_manager._check_suspicious_processes(mock_container)

        assert len(violations) == 1
        assert violations[0].violation_type == SecurityViolationType.PROCESS_VIOLATION
        assert violations[0].details["blocked_command"] == "rm"

    @pytest.mark.asyncio
    async def test_terminate_container(self, security_manager, mock_container):
        """Test container termination due to violations."""
        with patch.object(security_manager, "stop_monitoring") as mock_stop:
            await security_manager._terminate_container(mock_container, "Test reason")

            mock_stop.assert_called_once_with(mock_container.id)
            mock_container.stop.assert_called_once_with(timeout=5)
            mock_container.remove.assert_called_once_with(force=True)

            # Check violation was logged
            assert len(security_manager.violations) == 1
            assert security_manager.violations[0].details["reason"] == "Test reason"

    @pytest.mark.asyncio
    async def test_get_security_report(self, security_manager, mock_container):
        """Test generating security report."""
        # Add some test data
        violation = SecurityViolation(
            violation_id="test_violation",
            timestamp=asyncio.get_event_loop().time(),
            violation_type=SecurityViolationType.RESOURCE_LIMIT_EXCEEDED,
            severity=SecurityLevel.HIGH,
            container_id=mock_container.id,
            details={"resource": "cpu"},
            action_taken="logged",
        )
        security_manager.violations.append(violation)

        report = await security_manager.get_security_report(mock_container.id)

        assert report["container_id"] == mock_container.id
        assert report["total_violations"] == 1
        assert "resource_limit_exceeded" in report["violation_types"]
        assert report["policy_summary"]["network_policy"] == "none"

    @pytest.mark.asyncio
    async def test_cleanup(self, security_manager, mock_container):
        """Test security manager cleanup."""
        # Add some test data
        security_manager.monitored_containers[mock_container.id] = mock_container
        mock_task = AsyncMock()
        security_manager.monitoring_tasks[mock_container.id] = mock_task
        security_manager.violations.append(MagicMock())
        security_manager.metrics_history.append(MagicMock())

        await security_manager.cleanup()

        mock_task.cancel.assert_called_once()
        assert len(security_manager.violations) == 0
        assert len(security_manager.metrics_history) == 0
        assert len(security_manager.monitored_containers) == 0


class TestSecurityPolicyConfiguration:
    """Test cases for security policy configuration."""

    def test_default_security_policy(self):
        """Test default security policy values."""
        policy = SecurityPolicy()

        assert policy.resource_limits.memory_mb == 512
        assert policy.resource_limits.cpu_percent == 50.0
        assert policy.network_policy == NetworkPolicy.NONE
        assert "rm" in policy.blocked_commands
        assert ".txt" in policy.allowed_file_extensions
        assert "/etc/" in policy.blocked_file_patterns

    def test_custom_security_policy(self):
        """Test custom security policy configuration."""
        custom_limits = ResourceLimits(
            memory_mb=1024, cpu_percent=75.0, max_processes=20
        )

        policy = SecurityPolicy(
            resource_limits=custom_limits,
            network_policy=NetworkPolicy.RESTRICTED,
            violation_threshold=10,
        )

        assert policy.resource_limits.memory_mb == 1024
        assert policy.resource_limits.cpu_percent == 75.0
        assert policy.network_policy == NetworkPolicy.RESTRICTED
        assert policy.violation_threshold == 10


class TestResourceLimits:
    """Test cases for resource limits."""

    def test_default_resource_limits(self):
        """Test default resource limit values."""
        limits = ResourceLimits()

        assert limits.memory_mb == 512
        assert limits.cpu_percent == 50.0
        assert limits.disk_mb == 1024
        assert limits.max_processes == 50
        assert limits.max_open_files == 1024
        assert limits.max_network_connections == 10
        assert limits.execution_timeout == 300

    def test_custom_resource_limits(self):
        """Test custom resource limit configuration."""
        limits = ResourceLimits(
            memory_mb=256, cpu_percent=25.0, max_processes=10, execution_timeout=60
        )

        assert limits.memory_mb == 256
        assert limits.cpu_percent == 25.0
        assert limits.max_processes == 10
        assert limits.execution_timeout == 60


if __name__ == "__main__":
    pytest.main(["-v", __file__])
