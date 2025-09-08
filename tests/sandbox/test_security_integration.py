"""
Integration tests for enhanced sandbox security.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from app.config import SandboxSettings
from app.sandbox.core.manager import SandboxManager
from app.sandbox.core.sandbox import DockerSandbox
from app.sandbox.core.security import NetworkPolicy, ResourceLimits, SecurityPolicy


@pytest.fixture
def security_policy():
    """Create a test security policy."""
    return SecurityPolicy(
        resource_limits=ResourceLimits(
            memory_mb=256, cpu_percent=25.0, max_processes=5
        ),
        network_policy=NetworkPolicy.NONE,
        blocked_commands={"rm", "sudo"},
        allowed_file_extensions={".txt", ".py"},
        blocked_file_patterns={"/etc/"},
        enable_monitoring=True,
        violation_threshold=2,
        auto_terminate_on_violation=True,
    )


@pytest_asyncio.fixture
async def sandbox_manager(security_policy):
    """Create a sandbox manager with security policy."""
    manager = SandboxManager(
        max_sandboxes=2,
        idle_timeout=60,
        cleanup_interval=30,
        security_policy=security_policy,
    )
    try:
        yield manager
    finally:
        await manager.cleanup()


@pytest.fixture
def mock_docker_client():
    """Mock Docker client."""
    client = MagicMock()
    client.images.get.return_value = MagicMock()
    client.api.create_host_config.return_value = {}
    client.api.create_container.return_value = {"Id": "test_container_123"}

    mock_container = MagicMock()
    mock_container.id = "test_container_123"
    mock_container.start.return_value = None
    mock_container.stop.return_value = None
    mock_container.remove.return_value = None
    mock_container.update.return_value = None
    mock_container.exec_run.return_value = MagicMock(exit_code=0, output=b"success")

    client.containers.get.return_value = mock_container
    return client


class TestSandboxManagerSecurity:
    """Test cases for sandbox manager security integration."""

    @pytest.mark.asyncio
    async def test_create_sandbox_with_security_policy(
        self, sandbox_manager, mock_docker_client
    ):
        """Test creating sandbox with security policy applied."""
        with patch("docker.from_env", return_value=mock_docker_client), patch(
            "app.sandbox.core.terminal.AsyncDockerizedTerminal"
        ) as mock_terminal:

            mock_terminal_instance = AsyncMock()
            mock_terminal.return_value = mock_terminal_instance

            sandbox_id = await sandbox_manager.create_sandbox()

            assert sandbox_id in sandbox_manager._sandboxes
            # Verify security policy was applied
            mock_docker_client.containers.get().update.assert_called()

    @pytest.mark.asyncio
    async def test_validate_file_operation_allowed(self, sandbox_manager):
        """Test file operation validation for allowed files."""
        result = await sandbox_manager.validate_file_operation(
            "test_id", "/work/test.py"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_file_operation_blocked(self, sandbox_manager):
        """Test file operation validation for blocked files."""
        result = await sandbox_manager.validate_file_operation("test_id", "/etc/passwd")
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_command_execution_allowed(self, sandbox_manager):
        """Test command validation for allowed commands."""
        result = await sandbox_manager.validate_command_execution(
            "test_id", "python script.py"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_command_execution_blocked(self, sandbox_manager):
        """Test command validation for blocked commands."""
        result = await sandbox_manager.validate_command_execution("test_id", "rm -rf /")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_security_report(self, sandbox_manager):
        """Test getting security report."""
        report = await sandbox_manager.get_security_report()

        assert "report_timestamp" in report
        assert "total_violations" in report
        assert "policy_summary" in report
        assert report["policy_summary"]["network_policy"] == "none"

    @pytest.mark.asyncio
    async def test_get_stats_includes_security_info(self, sandbox_manager):
        """Test that stats include security information."""
        stats = sandbox_manager.get_stats()

        assert "security_violations" in stats
        assert "monitored_containers" in stats
        assert stats["security_violations"] == 0
        assert stats["monitored_containers"] == 0


class TestDockerSandboxSecurity:
    """Test cases for DockerSandbox security integration."""

    @pytest.fixture
    def mock_security_manager(self):
        """Mock security manager."""
        manager = AsyncMock()
        manager.validate_command_execution.return_value = True
        manager.validate_file_access.return_value = True
        return manager

    @pytest.fixture
    def sandbox_with_security(self, mock_security_manager):
        """Create sandbox with security manager."""
        config = SandboxSettings()
        sandbox = DockerSandbox(config, security_manager=mock_security_manager)

        # Mock container and terminal
        sandbox.container = MagicMock()
        sandbox.container.id = "test_container_123"
        sandbox.terminal = AsyncMock()

        return sandbox

    @pytest.mark.asyncio
    async def test_run_command_with_validation(self, sandbox_with_security):
        """Test command execution with security validation."""
        sandbox_with_security.terminal.run_command.return_value = "success"

        result = await sandbox_with_security.run_command("echo hello")

        assert result == "success"
        sandbox_with_security.security_manager.validate_command_execution.assert_called_once_with(
            "test_container_123", "echo hello"
        )

    @pytest.mark.asyncio
    async def test_run_command_blocked_by_security(self, sandbox_with_security):
        """Test command execution blocked by security policy."""
        sandbox_with_security.security_manager.validate_command_execution.return_value = (
            False
        )

        with pytest.raises(PermissionError, match="Command blocked by security policy"):
            await sandbox_with_security.run_command("rm -rf /")

    @pytest.mark.asyncio
    async def test_read_file_with_validation(self, sandbox_with_security):
        """Test file reading with security validation."""
        # Mock file reading
        with patch.object(
            sandbox_with_security, "_safe_resolve_path", return_value="/work/test.txt"
        ), patch("asyncio.to_thread") as mock_thread, patch.object(
            sandbox_with_security, "_read_from_tar", return_value=b"file content"
        ):

            mock_thread.return_value = (MagicMock(), None)

            result = await sandbox_with_security.read_file("test.txt")

            assert result == "file content"
            sandbox_with_security.security_manager.validate_file_access.assert_called_once_with(
                "test_container_123", "test.txt"
            )

    @pytest.mark.asyncio
    async def test_read_file_blocked_by_security(self, sandbox_with_security):
        """Test file reading blocked by security policy."""
        sandbox_with_security.security_manager.validate_file_access.return_value = False

        with pytest.raises(
            PermissionError, match="File access blocked by security policy"
        ):
            await sandbox_with_security.read_file("/etc/passwd")

    @pytest.mark.asyncio
    async def test_write_file_with_validation(self, sandbox_with_security):
        """Test file writing with security validation."""
        # Mock file writing
        with patch.object(
            sandbox_with_security, "_safe_resolve_path", return_value="/work/test.txt"
        ), patch.object(sandbox_with_security, "run_command"), patch.object(
            sandbox_with_security, "_create_tar_stream"
        ), patch(
            "asyncio.to_thread"
        ):

            await sandbox_with_security.write_file("test.txt", "content")

            sandbox_with_security.security_manager.validate_file_access.assert_called_once_with(
                "test_container_123", "test.txt"
            )

    @pytest.mark.asyncio
    async def test_write_file_blocked_by_security(self, sandbox_with_security):
        """Test file writing blocked by security policy."""
        sandbox_with_security.security_manager.validate_file_access.return_value = False

        with pytest.raises(
            PermissionError, match="File access blocked by security policy"
        ):
            await sandbox_with_security.write_file("/etc/passwd", "malicious content")


class TestSecurityPolicyEnforcement:
    """Test cases for security policy enforcement."""

    @pytest.mark.asyncio
    async def test_network_isolation_enforcement(self):
        """Test network isolation is properly enforced."""
        policy = SecurityPolicy(network_policy=NetworkPolicy.NONE)
        manager = SandboxManager(security_policy=policy)

        # Mock container with network connections
        mock_container = MagicMock()
        mock_container.attrs = {
            "NetworkSettings": {"Networks": {"bridge": {}, "custom_network": {}}}
        }

        mock_network = MagicMock()
        mock_container.client.networks.get.return_value = mock_network

        await manager.security_manager._apply_network_policy(mock_container)

        # Verify networks were disconnected
        assert mock_container.client.networks.get.call_count >= 1
        assert mock_network.disconnect.call_count >= 1

    @pytest.mark.asyncio
    async def test_resource_limits_enforcement(self):
        """Test resource limits are properly enforced."""
        limits = ResourceLimits(memory_mb=128, cpu_percent=10.0, max_processes=3)
        policy = SecurityPolicy(resource_limits=limits)
        manager = SandboxManager(security_policy=policy)

        mock_container = MagicMock()

        await manager.security_manager._apply_resource_limits(mock_container)

        # Verify container.update was called with correct limits
        mock_container.update.assert_called_once()
        call_args = mock_container.update.call_args[1]
        assert call_args["mem_limit"] == "128m"
        assert call_args["cpu_quota"] == 10000  # 10% of 100000
        assert call_args["pids_limit"] == 3

    @pytest.mark.asyncio
    async def test_filesystem_restrictions_enforcement(self):
        """Test filesystem restrictions are properly enforced."""
        policy = SecurityPolicy()
        manager = SandboxManager(security_policy=policy)

        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(exit_code=0)

        await manager.security_manager._apply_filesystem_restrictions(mock_container)

        # Verify sensitive directories were made read-only
        exec_calls = [call[0][0] for call in mock_container.exec_run.call_args_list]
        assert any("chmod -R a-w /etc" in call for call in exec_calls)
        assert any("mkdir -p /sandbox/work" in call for call in exec_calls)


class TestSecurityMonitoring:
    """Test cases for security monitoring."""

    @pytest.mark.asyncio
    async def test_monitoring_detects_violations(self):
        """Test that monitoring detects security violations."""
        policy = SecurityPolicy(
            resource_limits=ResourceLimits(memory_mb=100, cpu_percent=10.0),
            violation_threshold=1,
            auto_terminate_on_violation=False,  # Don't terminate for testing
        )
        manager = SandboxManager(security_policy=policy)

        mock_container = MagicMock()
        mock_container.id = "test_container"
        mock_container.stats.return_value = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 2000000},
                "system_cpu_usage": 10000000,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 1000000},
                "system_cpu_usage": 9000000,
            },
            "memory_stats": {"usage": 200 * 1024 * 1024},  # 200MB - exceeds 100MB limit
        }
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=b"2\n")

        # Collect metrics and check violations
        metrics = await manager.security_manager._collect_metrics(mock_container)
        violations = await manager.security_manager._check_violations(
            mock_container, metrics
        )

        # Should detect memory violation
        assert len(violations) > 0
        memory_violations = [
            v for v in violations if v.details.get("resource") == "memory"
        ]
        assert len(memory_violations) == 1

    @pytest.mark.asyncio
    async def test_monitoring_task_lifecycle(self):
        """Test monitoring task lifecycle management."""
        policy = SecurityPolicy(enable_monitoring=True)
        manager = SandboxManager(security_policy=policy)

        mock_container = MagicMock()
        mock_container.id = "test_container"

        # Start monitoring
        with patch.object(
            manager.security_manager, "_monitor_container"
        ) as mock_monitor:
            mock_monitor.return_value = AsyncMock()

            await manager.security_manager.start_monitoring(mock_container)

            assert mock_container.id in manager.security_manager.monitored_containers
            assert mock_container.id in manager.security_manager.monitoring_tasks

        # Stop monitoring
        await manager.security_manager.stop_monitoring(mock_container.id)

        assert mock_container.id not in manager.security_manager.monitored_containers
        assert mock_container.id not in manager.security_manager.monitoring_tasks


if __name__ == "__main__":
    pytest.main(["-v", __file__])
