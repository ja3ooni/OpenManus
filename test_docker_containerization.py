#!/usr/bin/env python3
"""
Test script for Docker containerization implementation.

This script tests the Docker containerization features including:
- Multi-stage builds
- Health check endpoints
- Signal handling and graceful shutdown
- Container monitoring capabilities
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import pytest


class DockerContainerizationTest:
    """Test Docker containerization implementation"""

    def __init__(self):
        self.test_results = []
        self.container_id = None
        self.base_url = "http://localhost:8080"

    def log_result(self, test_name: str, success: bool, message: str = ""):
        """Log test result"""
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
        if message:
            print(f"    {message}")

        self.test_results.append(
            {"test": test_name, "success": success, "message": message}
        )

    def test_dockerfile_structure(self):
        """Test Dockerfile structure and multi-stage builds"""
        print("\n=== Testing Dockerfile Structure ===")

        dockerfile_path = Path("Dockerfile")
        if not dockerfile_path.exists():
            self.log_result("Dockerfile exists", False, "Dockerfile not found")
            return

        content = dockerfile_path.read_text()

        # Test multi-stage build
        has_builder_stage = "FROM python:3.12-slim as builder" in content
        self.log_result("Multi-stage build (builder stage)", has_builder_stage)

        has_production_stage = "FROM python:3.12-slim as production" in content
        self.log_result("Multi-stage build (production stage)", has_production_stage)

        has_development_stage = "FROM builder as development" in content
        self.log_result("Multi-stage build (development stage)", has_development_stage)

        # Test security features
        has_nonroot_user = "useradd -r -g openmanus" in content
        self.log_result("Non-root user creation", has_nonroot_user)

        has_user_switch = "USER openmanus" in content
        self.log_result("User switch to non-root", has_user_switch)

        # Test health check
        has_healthcheck = "HEALTHCHECK" in content
        self.log_result("Health check configuration", has_healthcheck)

        # Test proper labels
        has_labels = "LABEL maintainer" in content
        self.log_result("Container labels", has_labels)

        # Test environment variables
        has_env_vars = "ENV PYTHONPATH=/app" in content
        self.log_result("Environment variables", has_env_vars)

        # Test exposed ports
        has_exposed_ports = "EXPOSE 8000 8080 9090" in content
        self.log_result("Exposed ports", has_exposed_ports)

    def test_docker_compose_structure(self):
        """Test Docker Compose configuration"""
        print("\n=== Testing Docker Compose Structure ===")

        # Test main docker-compose.yml
        compose_path = Path("docker-compose.yml")
        if not compose_path.exists():
            self.log_result("docker-compose.yml exists", False)
            return

        content = compose_path.read_text()

        # Test service definitions
        has_openmanus_service = "openmanus:" in content
        self.log_result("OpenManus service defined", has_openmanus_service)

        has_redis_service = "redis:" in content
        self.log_result("Redis service defined", has_redis_service)

        has_nginx_service = "nginx:" in content
        self.log_result("Nginx service defined", has_nginx_service)

        # Test health checks
        has_healthcheck = "healthcheck:" in content
        self.log_result("Health check configuration", has_healthcheck)

        # Test volumes
        has_volumes = "volumes:" in content
        self.log_result("Volume configuration", has_volumes)

        # Test networks
        has_networks = "networks:" in content
        self.log_result("Network configuration", has_networks)

        # Test development override
        dev_compose_path = Path("docker-compose.dev.yml")
        has_dev_override = dev_compose_path.exists()
        self.log_result("Development override exists", has_dev_override)

        # Test production override
        prod_compose_path = Path("docker-compose.prod.yml")
        has_prod_override = prod_compose_path.exists()
        self.log_result("Production override exists", has_prod_override)

    def test_build_docker_image(self):
        """Test building Docker image"""
        print("\n=== Testing Docker Image Build ===")

        try:
            # Build production image
            print("Building production Docker image...")
            # Use UTF-8 encoding for subprocess on Windows
            encoding = "utf-8" if sys.platform == "win32" else None

            result = subprocess.run(
                [
                    "docker",
                    "build",
                    "--target",
                    "production",
                    "-t",
                    "openmanus:test-production",
                    ".",
                ],
                capture_output=True,
                text=True,
                timeout=300,
                encoding=encoding,
                errors="replace",
            )

            if result.returncode == 0:
                self.log_result("Production image build", True)
            else:
                # Truncate error message to avoid encoding issues
                error_msg = (
                    result.stderr[:500] + "..."
                    if len(result.stderr) > 500
                    else result.stderr
                )
                self.log_result("Production image build", False, error_msg)
                return

            # Build development image
            print("Building development Docker image...")
            result = subprocess.run(
                [
                    "docker",
                    "build",
                    "--target",
                    "development",
                    "-t",
                    "openmanus:test-development",
                    ".",
                ],
                capture_output=True,
                text=True,
                timeout=300,
                encoding=encoding,
                errors="replace",
            )

            if result.returncode == 0:
                self.log_result("Development image build", True)
            else:
                error_msg = (
                    result.stderr[:500] + "..."
                    if len(result.stderr) > 500
                    else result.stderr
                )
                self.log_result("Development image build", False, error_msg)

        except subprocess.TimeoutExpired:
            self.log_result("Docker image build", False, "Build timeout")
        except Exception as e:
            self.log_result("Docker image build", False, str(e))

    def test_container_startup(self):
        """Test container startup and health checks"""
        print("\n=== Testing Container Startup ===")

        try:
            # Start container
            print("Starting Docker container...")
            # Use UTF-8 encoding for subprocess on Windows
            encoding = "utf-8" if sys.platform == "win32" else None

            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    "openmanus-test",
                    "-p",
                    "8080:8080",
                    "-p",
                    "9090:9090",
                    "openmanus:test-production",
                ],
                capture_output=True,
                text=True,
                encoding=encoding,
                errors="replace",
            )

            if result.returncode != 0:
                self.log_result("Container startup", False, result.stderr)
                return

            self.container_id = result.stdout.strip()
            self.log_result(
                "Container startup", True, f"Container ID: {self.container_id[:12]}"
            )

            # Wait for container to be ready
            print("Waiting for container to be ready...")
            time.sleep(10)

            # Check if container is running
            result = subprocess.run(
                ["docker", "ps", "-q", "-f", f"id={self.container_id}"],
                capture_output=True,
                text=True,
            )

            is_running = bool(result.stdout.strip())
            self.log_result("Container running", is_running)

            if not is_running:
                # Get container logs for debugging
                logs_result = subprocess.run(
                    ["docker", "logs", self.container_id],
                    capture_output=True,
                    text=True,
                )
                self.log_result(
                    "Container logs", False, logs_result.stdout + logs_result.stderr
                )

        except Exception as e:
            self.log_result("Container startup", False, str(e))

    async def test_health_endpoints(self):
        """Test health check endpoints"""
        print("\n=== Testing Health Endpoints ===")

        if not self.container_id:
            self.log_result("Health endpoints", False, "No container running")
            return

        try:
            async with aiohttp.ClientSession() as session:
                # Test liveness probe
                try:
                    async with session.get(
                        f"{self.base_url}/health/live", timeout=10
                    ) as response:
                        liveness_ok = response.status == 200
                        self.log_result(
                            "Liveness probe", liveness_ok, f"Status: {response.status}"
                        )
                except Exception as e:
                    self.log_result("Liveness probe", False, str(e))

                # Test readiness probe
                try:
                    async with session.get(
                        f"{self.base_url}/health/ready", timeout=10
                    ) as response:
                        readiness_ok = response.status in [
                            200,
                            503,
                        ]  # 503 is acceptable if not ready
                        response_data = await response.json()
                        self.log_result(
                            "Readiness probe",
                            readiness_ok,
                            f"Status: {response.status}, Ready: {response_data.get('ready', False)}",
                        )
                except Exception as e:
                    self.log_result("Readiness probe", False, str(e))

                # Test comprehensive health check
                try:
                    async with session.get(
                        f"{self.base_url}/health", timeout=15
                    ) as response:
                        health_ok = response.status in [200, 503]
                        response_data = await response.json()
                        self.log_result(
                            "Comprehensive health check",
                            health_ok,
                            f"Status: {response.status}, Health: {response_data.get('status', 'unknown')}",
                        )
                except Exception as e:
                    self.log_result("Comprehensive health check", False, str(e))

                # Test metrics endpoint
                try:
                    async with session.get(
                        f"{self.base_url}/metrics", timeout=10
                    ) as response:
                        metrics_ok = response.status == 200
                        content_type = response.headers.get("content-type", "")
                        is_prometheus_format = "text/plain" in content_type
                        self.log_result(
                            "Metrics endpoint",
                            metrics_ok and is_prometheus_format,
                            f"Status: {response.status}, Content-Type: {content_type}",
                        )
                except Exception as e:
                    self.log_result("Metrics endpoint", False, str(e))

                # Test status endpoint
                try:
                    async with session.get(
                        f"{self.base_url}/status", timeout=10
                    ) as response:
                        status_ok = response.status == 200
                        response_data = await response.json()
                        has_application_info = "application" in response_data
                        self.log_result(
                            "Status endpoint",
                            status_ok and has_application_info,
                            f"Status: {response.status}, Has app info: {has_application_info}",
                        )
                except Exception as e:
                    self.log_result("Status endpoint", False, str(e))

        except Exception as e:
            self.log_result("Health endpoints", False, str(e))

    def test_graceful_shutdown(self):
        """Test graceful shutdown handling"""
        print("\n=== Testing Graceful Shutdown ===")

        if not self.container_id:
            self.log_result("Graceful shutdown", False, "No container running")
            return

        try:
            # Send SIGTERM to container
            print("Sending SIGTERM to container...")
            result = subprocess.run(
                ["docker", "kill", "-s", "TERM", self.container_id],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                self.log_result("Send SIGTERM", False, result.stderr)
                return

            self.log_result("Send SIGTERM", True)

            # Wait for graceful shutdown
            print("Waiting for graceful shutdown...")
            time.sleep(5)

            # Check if container stopped gracefully
            result = subprocess.run(
                ["docker", "ps", "-q", "-f", f"id={self.container_id}"],
                capture_output=True,
                text=True,
            )

            container_stopped = not bool(result.stdout.strip())
            self.log_result("Graceful shutdown", container_stopped)

            if not container_stopped:
                # Force stop if graceful shutdown failed
                subprocess.run(
                    ["docker", "stop", self.container_id], capture_output=True
                )

        except Exception as e:
            self.log_result("Graceful shutdown", False, str(e))

    def test_docker_compose_functionality(self):
        """Test Docker Compose functionality"""
        print("\n=== Testing Docker Compose Functionality ===")

        try:
            # Test docker-compose config validation
            result = subprocess.run(
                ["docker-compose", "config"], capture_output=True, text=True
            )

            config_valid = result.returncode == 0
            self.log_result(
                "Docker Compose config validation",
                config_valid,
                result.stderr if not config_valid else "Config is valid",
            )

            # Test development override
            result = subprocess.run(
                [
                    "docker-compose",
                    "-f",
                    "docker-compose.yml",
                    "-f",
                    "docker-compose.dev.yml",
                    "config",
                ],
                capture_output=True,
                text=True,
            )

            dev_config_valid = result.returncode == 0
            self.log_result("Development config validation", dev_config_valid)

            # Test production override
            result = subprocess.run(
                [
                    "docker-compose",
                    "-f",
                    "docker-compose.yml",
                    "-f",
                    "docker-compose.prod.yml",
                    "config",
                ],
                capture_output=True,
                text=True,
            )

            prod_config_valid = result.returncode == 0
            self.log_result("Production config validation", prod_config_valid)

        except Exception as e:
            self.log_result("Docker Compose functionality", False, str(e))

    def cleanup(self):
        """Clean up test resources"""
        print("\n=== Cleaning Up ===")

        if self.container_id:
            try:
                # Stop and remove container
                subprocess.run(
                    ["docker", "stop", self.container_id],
                    capture_output=True,
                    timeout=30,
                )
                subprocess.run(["docker", "rm", self.container_id], capture_output=True)
                print(f"Cleaned up container {self.container_id[:12]}")
            except Exception as e:
                print(f"Error cleaning up container: {e}")

        # Remove test images
        try:
            subprocess.run(
                ["docker", "rmi", "openmanus:test-production"], capture_output=True
            )
            subprocess.run(
                ["docker", "rmi", "openmanus:test-development"], capture_output=True
            )
            print("Cleaned up test images")
        except Exception as e:
            print(f"Error cleaning up images: {e}")

    async def run_all_tests(self):
        """Run all Docker containerization tests"""
        print("🐳 Docker Containerization Test Suite")
        print("=" * 50)

        # Run tests in order
        self.test_dockerfile_structure()
        self.test_docker_compose_structure()
        self.test_build_docker_image()
        self.test_container_startup()
        await self.test_health_endpoints()
        self.test_graceful_shutdown()
        self.test_docker_compose_functionality()

        # Print summary
        print("\n" + "=" * 50)
        print("📊 Test Summary")
        print("=" * 50)

        passed = sum(1 for result in self.test_results if result["success"])
        total = len(self.test_results)

        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success rate: {(passed/total)*100:.1f}%")

        if passed == total:
            print("\n🎉 All Docker containerization tests passed!")
            return True
        else:
            print(f"\n❌ {total - passed} tests failed")
            print("\nFailed tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result['message']}")
            return False


async def main():
    """Main test function"""
    tester = DockerContainerizationTest()

    try:
        success = await tester.run_all_tests()
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 1

    except Exception as e:
        print(f"\n\nTest suite error: {e}")
        return 1

    finally:
        tester.cleanup()


if __name__ == "__main__":
    # Check if Docker is available
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Docker is not available")
            sys.exit(1)
    except FileNotFoundError:
        print("❌ Docker is not installed")
        sys.exit(1)

    # Check if docker-compose is available
    try:
        result = subprocess.run(
            ["docker-compose", "--version"], capture_output=True, text=True
        )
        if result.returncode != 0:
            print("❌ Docker Compose is not available")
            sys.exit(1)
    except FileNotFoundError:
        print("❌ Docker Compose is not installed")
        sys.exit(1)

    # Run tests
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
