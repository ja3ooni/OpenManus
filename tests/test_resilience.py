"""
Tests for resilience patterns including retry logic and circuit breaker.
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.exceptions import (
    CircuitBreakerOpenError,
    ErrorClassification,
    ErrorContext,
    NetworkError,
    OpenManusError,
)
from app.resilience import (
    CircuitBreaker,
    CircuitState,
    RetryManager,
    RetryStrategy,
    circuit_breaker,
    get_circuit_breaker,
    resilient,
    retry_async,
    retry_sync,
)


class TestRetryManager:
    """Test cases for RetryManager"""

    def test_init_default_values(self):
        """Test RetryManager initialization with default values"""
        retry_manager = RetryManager()
        assert retry_manager.max_retries == 3
        assert retry_manager.base_delay == 1.0
        assert retry_manager.max_delay == 60.0
        assert retry_manager.backoff_multiplier == 2.0
        assert retry_manager.jitter is True
        assert retry_manager.strategy == RetryStrategy.EXPONENTIAL_BACKOFF

    def test_calculate_delay_exponential(self):
        """Test exponential backoff delay calculation"""
        retry_manager = RetryManager(
            base_delay=1.0, backoff_multiplier=2.0, jitter=False
        )

        assert retry_manager._calculate_delay(1) == 1.0
        assert retry_manager._calculate_delay(2) == 2.0
        assert retry_manager._calculate_delay(3) == 4.0
        assert retry_manager._calculate_delay(4) == 8.0

    def test_calculate_delay_linear(self):
        """Test linear backoff delay calculation"""
        retry_manager = RetryManager(
            base_delay=2.0, strategy=RetryStrategy.LINEAR_BACKOFF, jitter=False
        )

        assert retry_manager._calculate_delay(1) == 2.0
        assert retry_manager._calculate_delay(2) == 4.0
        assert retry_manager._calculate_delay(3) == 6.0

    def test_calculate_delay_fixed(self):
        """Test fixed delay calculation"""
        retry_manager = RetryManager(
            base_delay=3.0, strategy=RetryStrategy.FIXED_DELAY, jitter=False
        )

        assert retry_manager._calculate_delay(1) == 3.0
        assert retry_manager._calculate_delay(2) == 3.0
        assert retry_manager._calculate_delay(3) == 3.0

    def test_calculate_delay_immediate(self):
        """Test immediate retry (no delay)"""
        retry_manager = RetryManager(strategy=RetryStrategy.IMMEDIATE)

        assert retry_manager._calculate_delay(1) == 0
        assert retry_manager._calculate_delay(2) == 0

    def test_calculate_delay_max_limit(self):
        """Test delay respects maximum limit"""
        retry_manager = RetryManager(
            base_delay=10.0, max_delay=15.0, backoff_multiplier=2.0, jitter=False
        )

        assert retry_manager._calculate_delay(1) == 10.0
        assert retry_manager._calculate_delay(2) == 15.0  # Capped at max_delay
        assert retry_manager._calculate_delay(3) == 15.0  # Still capped

    def test_should_retry_recoverable_error(self):
        """Test retry decision for recoverable errors"""
        retry_manager = RetryManager(max_retries=3)

        error = OpenManusError(
            "Test error", classification=ErrorClassification.TRANSIENT, recoverable=True
        )

        assert retry_manager._should_retry(error, 1) is True
        assert retry_manager._should_retry(error, 2) is True
        assert retry_manager._should_retry(error, 3) is False  # Max retries reached

    def test_should_retry_permanent_error(self):
        """Test retry decision for permanent errors"""
        retry_manager = RetryManager(max_retries=3)

        error = OpenManusError(
            "Test error",
            classification=ErrorClassification.PERMANENT,
            recoverable=False,
        )

        assert retry_manager._should_retry(error, 1) is False

    def test_should_retry_standard_exceptions(self):
        """Test retry decision for standard exceptions"""
        retry_manager = RetryManager(max_retries=3)

        assert retry_manager._should_retry(ConnectionError("Test"), 1) is True
        assert retry_manager._should_retry(TimeoutError("Test"), 1) is True
        assert retry_manager._should_retry(OSError("Test"), 1) is True
        assert retry_manager._should_retry(ValueError("Test"), 1) is False

    @pytest.mark.asyncio
    async def test_retry_async_success_first_attempt(self):
        """Test async retry with success on first attempt"""
        retry_manager = RetryManager()
        mock_operation = AsyncMock(return_value="success")

        result = await retry_manager.retry_async(mock_operation)

        assert result == "success"
        assert mock_operation.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_async_success_after_retries(self):
        """Test async retry with success after failures"""
        retry_manager = RetryManager(max_retries=3, base_delay=0.01)
        mock_operation = AsyncMock(
            side_effect=[NetworkError("Fail"), NetworkError("Fail"), "success"]
        )

        result = await retry_manager.retry_async(mock_operation)

        assert result == "success"
        assert mock_operation.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_async_permanent_failure(self):
        """Test async retry with permanent failure"""
        retry_manager = RetryManager(max_retries=3)
        error = OpenManusError(
            "Permanent error", classification=ErrorClassification.PERMANENT
        )
        mock_operation = AsyncMock(side_effect=error)

        with pytest.raises(OpenManusError):
            await retry_manager.retry_async(mock_operation)

        assert mock_operation.call_count == 1  # No retries for permanent errors

    def test_retry_sync_success_first_attempt(self):
        """Test sync retry with success on first attempt"""
        retry_manager = RetryManager()
        mock_operation = Mock(return_value="success")

        result = retry_manager.retry_sync(mock_operation)

        assert result == "success"
        assert mock_operation.call_count == 1

    def test_retry_sync_success_after_retries(self):
        """Test sync retry with success after failures"""
        retry_manager = RetryManager(max_retries=3, base_delay=0.01)
        mock_operation = Mock(
            side_effect=[NetworkError("Fail"), NetworkError("Fail"), "success"]
        )

        result = retry_manager.retry_sync(mock_operation)

        assert result == "success"
        assert mock_operation.call_count == 3


class TestCircuitBreaker:
    """Test cases for CircuitBreaker"""

    def test_init_default_values(self):
        """Test CircuitBreaker initialization"""
        cb = CircuitBreaker("test-service")

        assert cb.service_name == "test-service"
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 60
        assert cb.success_threshold == 3
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0

    @pytest.mark.asyncio
    async def test_call_async_success(self):
        """Test successful async call through circuit breaker"""
        cb = CircuitBreaker("test-service")
        mock_func = AsyncMock(return_value="success")

        result = await cb.call_async(mock_func)

        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_call_async_failure_below_threshold(self):
        """Test async failures below threshold"""
        cb = CircuitBreaker("test-service", failure_threshold=3)
        mock_func = AsyncMock(side_effect=Exception("Test error"))

        # First two failures should not open circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await cb.call_async(mock_func)

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 2

    @pytest.mark.asyncio
    async def test_call_async_failure_opens_circuit(self):
        """Test async failures that open the circuit"""
        cb = CircuitBreaker("test-service", failure_threshold=2)
        mock_func = AsyncMock(side_effect=Exception("Test error"))

        # Failures that should open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await cb.call_async(mock_func)

        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 2

        # Next call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await cb.call_async(mock_func)

    @pytest.mark.asyncio
    async def test_call_async_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state"""
        cb = CircuitBreaker(
            "test-service",
            failure_threshold=1,
            recovery_timeout=0.01,  # Very short timeout for testing
            success_threshold=2,
        )

        # Open the circuit
        mock_func = AsyncMock(side_effect=Exception("Test error"))
        with pytest.raises(Exception):
            await cb.call_async(mock_func)

        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.02)

        # First call after timeout should transition to half-open
        mock_func = AsyncMock(return_value="success")
        result = await cb.call_async(mock_func)

        assert result == "success"
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.success_count == 1

        # Another success should close the circuit
        result = await cb.call_async(mock_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.success_count == 0

    def test_call_sync_success(self):
        """Test successful sync call through circuit breaker"""
        cb = CircuitBreaker("test-service")
        mock_func = Mock(return_value="success")

        result = cb.call_sync(mock_func)

        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_call_sync_failure_opens_circuit(self):
        """Test sync failures that open the circuit"""
        cb = CircuitBreaker("test-service", failure_threshold=2)
        mock_func = Mock(side_effect=Exception("Test error"))

        # Failures that should open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                cb.call_sync(mock_func)

        assert cb.state == CircuitState.OPEN

        # Next call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            cb.call_sync(mock_func)


class TestDecorators:
    """Test cases for resilience decorators"""

    @pytest.mark.asyncio
    async def test_retry_async_decorator(self):
        """Test async retry decorator"""
        call_count = 0

        @retry_async(max_retries=2, base_delay=0.01)
        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Temporary failure")
            return "success"

        result = await failing_function()

        assert result == "success"
        assert call_count == 3

    def test_retry_sync_decorator(self):
        """Test sync retry decorator"""
        call_count = 0

        @retry_sync(max_retries=2, base_delay=0.01)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Temporary failure")
            return "success"

        result = failing_function()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator_async(self):
        """Test async circuit breaker decorator"""

        @circuit_breaker("test-service", failure_threshold=2)
        async def failing_function():
            raise Exception("Always fails")

        # First two calls should fail normally
        for _ in range(2):
            with pytest.raises(Exception):
                await failing_function()

        # Third call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await failing_function()

    def test_circuit_breaker_decorator_sync(self):
        """Test sync circuit breaker decorator"""

        @circuit_breaker("test-service-sync", failure_threshold=2)
        def failing_function():
            raise Exception("Always fails")

        # First two calls should fail normally
        for _ in range(2):
            with pytest.raises(Exception):
                failing_function()

        # Third call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            failing_function()

    @pytest.mark.asyncio
    async def test_resilient_decorator_async(self):
        """Test combined resilient decorator for async functions"""
        call_count = 0

        @resilient(
            "test-resilient", max_retries=2, base_delay=0.01, failure_threshold=5
        )
        async def sometimes_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Temporary failure")
            return "success"

        result = await sometimes_failing_function()

        assert result == "success"
        assert call_count == 3

    def test_resilient_decorator_sync(self):
        """Test combined resilient decorator for sync functions"""
        call_count = 0

        @resilient(
            "test-resilient-sync", max_retries=2, base_delay=0.01, failure_threshold=5
        )
        def sometimes_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Temporary failure")
            return "success"

        result = sometimes_failing_function()

        assert result == "success"
        assert call_count == 3


class TestCircuitBreakerRegistry:
    """Test cases for circuit breaker registry"""

    def test_get_circuit_breaker_creates_new(self):
        """Test that get_circuit_breaker creates new instances"""
        cb1 = get_circuit_breaker("service1")
        cb2 = get_circuit_breaker("service2")

        assert cb1 is not cb2
        assert cb1.service_name == "service1"
        assert cb2.service_name == "service2"

    def test_get_circuit_breaker_reuses_existing(self):
        """Test that get_circuit_breaker reuses existing instances"""
        cb1 = get_circuit_breaker("service3")
        cb2 = get_circuit_breaker("service3")

        assert cb1 is cb2
        assert cb1.service_name == "service3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
