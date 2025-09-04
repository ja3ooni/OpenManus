"""
Resilience patterns for OpenManus including retry logic and circuit breaker implementation.
"""

import asyncio
import functools
import logging
import random
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type, Union
from uuid import uuid4

from .exceptions import (
    CircuitBreakerError,
    CircuitBreakerOpenError,
    ErrorClassification,
    ErrorContext,
    NetworkError,
    OpenManusError,
    TimeoutError,
)

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class RetryStrategy(Enum):
    """Retry strategies"""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"


class RetryManager:
    """Intelligent retry logic with configurable exponential backoff"""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_multiplier: float = 2.0,
        jitter: bool = True,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
        self.strategy = strategy

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt"""
        if self.strategy == RetryStrategy.IMMEDIATE:
            return 0
        elif self.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * attempt
        else:  # EXPONENTIAL_BACKOFF
            delay = self.base_delay * (self.backoff_multiplier ** (attempt - 1))

        # Apply maximum delay limit
        delay = min(delay, self.max_delay)

        # Add jitter to prevent thundering herd
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)

        return delay

    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if the exception should trigger a retry"""
        if attempt >= self.max_retries:
            return False

        # Don't retry permanent errors
        if isinstance(exception, OpenManusError):
            return exception.recoverable and exception.classification in [
                ErrorClassification.TRANSIENT,
                ErrorClassification.RESOURCE,
            ]

        # Retry common transient exceptions
        return isinstance(
            exception, (TimeoutError, NetworkError, ConnectionError, OSError)
        )

    async def retry_async(
        self,
        operation: Callable,
        *args,
        context: Optional[ErrorContext] = None,
        **kwargs,
    ) -> Any:
        """Execute async operation with retry logic"""
        last_exception = None

        for attempt in range(1, self.max_retries + 2):  # +1 for initial attempt
            try:
                if attempt > 1:
                    delay = self._calculate_delay(attempt - 1)
                    logger.info(
                        f"Retrying operation after {delay:.2f}s (attempt {attempt}/{self.max_retries + 1})",
                        extra={
                            "correlation_id": (
                                context.correlation_id if context else None
                            ),
                            "operation": context.operation if context else "unknown",
                            "attempt": attempt,
                            "delay": delay,
                        },
                    )
                    await asyncio.sleep(delay)

                result = await operation(*args, **kwargs)

                if attempt > 1:
                    logger.info(
                        f"Operation succeeded on attempt {attempt}",
                        extra={
                            "correlation_id": (
                                context.correlation_id if context else None
                            ),
                            "operation": context.operation if context else "unknown",
                            "attempt": attempt,
                        },
                    )

                return result

            except Exception as e:
                last_exception = e

                if not self._should_retry(e, attempt - 1):
                    logger.error(
                        f"Operation failed permanently: {str(e)}",
                        extra={
                            "correlation_id": (
                                context.correlation_id if context else None
                            ),
                            "operation": context.operation if context else "unknown",
                            "attempt": attempt,
                            "error": str(e),
                        },
                    )
                    break

                logger.warning(
                    f"Operation failed on attempt {attempt}: {str(e)}",
                    extra={
                        "correlation_id": context.correlation_id if context else None,
                        "operation": context.operation if context else "unknown",
                        "attempt": attempt,
                        "error": str(e),
                    },
                )

        # All retries exhausted
        raise last_exception

    def retry_sync(
        self,
        operation: Callable,
        *args,
        context: Optional[ErrorContext] = None,
        **kwargs,
    ) -> Any:
        """Execute sync operation with retry logic"""
        last_exception = None

        for attempt in range(1, self.max_retries + 2):
            try:
                if attempt > 1:
                    delay = self._calculate_delay(attempt - 1)
                    logger.info(
                        f"Retrying operation after {delay:.2f}s (attempt {attempt}/{self.max_retries + 1})",
                        extra={
                            "correlation_id": (
                                context.correlation_id if context else None
                            ),
                            "operation": context.operation if context else "unknown",
                            "attempt": attempt,
                            "delay": delay,
                        },
                    )
                    time.sleep(delay)

                result = operation(*args, **kwargs)

                if attempt > 1:
                    logger.info(
                        f"Operation succeeded on attempt {attempt}",
                        extra={
                            "correlation_id": (
                                context.correlation_id if context else None
                            ),
                            "operation": context.operation if context else "unknown",
                            "attempt": attempt,
                        },
                    )

                return result

            except Exception as e:
                last_exception = e

                if not self._should_retry(e, attempt - 1):
                    logger.error(
                        f"Operation failed permanently: {str(e)}",
                        extra={
                            "correlation_id": (
                                context.correlation_id if context else None
                            ),
                            "operation": context.operation if context else "unknown",
                            "attempt": attempt,
                            "error": str(e),
                        },
                    )
                    break

                logger.warning(
                    f"Operation failed on attempt {attempt}: {str(e)}",
                    extra={
                        "correlation_id": context.correlation_id if context else None,
                        "operation": context.operation if context else "unknown",
                        "attempt": attempt,
                        "error": str(e),
                    },
                )

        raise last_exception


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures"""

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 3,
        expected_exception: Type[Exception] = Exception,
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
        self._lock = asyncio.Lock()

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset"""
        if self.last_failure_time is None:
            return True

        return (
            datetime.now(timezone.utc) - self.last_failure_time
        ).total_seconds() >= self.recovery_timeout

    async def _on_success(self):
        """Handle successful operation"""
        async with self._lock:
            self.failure_count = 0

            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
                    logger.info(
                        f"Circuit breaker closed for service: {self.service_name}",
                        extra={"service": self.service_name, "state": self.state.value},
                    )

    async def _on_failure(self):
        """Handle failed operation"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now(timezone.utc)
            self.success_count = 0

            if (
                self.state == CircuitState.CLOSED
                and self.failure_count >= self.failure_threshold
            ):
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker opened for service: {self.service_name}",
                    extra={
                        "service": self.service_name,
                        "state": self.state.value,
                        "failure_count": self.failure_count,
                    },
                )
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker reopened for service: {self.service_name}",
                    extra={"service": self.service_name, "state": self.state.value},
                )

    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection"""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(
                        f"Circuit breaker half-open for service: {self.service_name}",
                        extra={"service": self.service_name, "state": self.state.value},
                    )
                else:
                    raise CircuitBreakerOpenError(self.service_name)

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except self.expected_exception as e:
            await self._on_failure()
            raise e

    def call_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute sync function with circuit breaker protection"""
        # Note: This is a simplified sync version
        # In production, you might want to use threading.Lock instead
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(
                    f"Circuit breaker half-open for service: {self.service_name}",
                    extra={"service": self.service_name, "state": self.state.value},
                )
            else:
                raise CircuitBreakerOpenError(self.service_name)

        try:
            result = func(*args, **kwargs)
            # Simplified success handling for sync version
            self.failure_count = 0
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
            return result
        except self.expected_exception as e:
            # Simplified failure handling for sync version
            self.failure_count += 1
            self.last_failure_time = datetime.now(timezone.utc)
            self.success_count = 0

            if (
                self.state == CircuitState.CLOSED
                and self.failure_count >= self.failure_threshold
            ):
                self.state = CircuitState.OPEN
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
            raise e


# Global circuit breakers registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    service_name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    success_threshold: int = 3,
    expected_exception: Type[Exception] = Exception,
) -> CircuitBreaker:
    """Get or create a circuit breaker for a service"""
    if service_name not in _circuit_breakers:
        _circuit_breakers[service_name] = CircuitBreaker(
            service_name=service_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            expected_exception=expected_exception,
        )
    return _circuit_breakers[service_name]


# Decorators for easy use
def retry_async(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
):
    """Decorator for async functions with retry logic"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            retry_manager = RetryManager(
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                strategy=strategy,
            )
            context = ErrorContext(operation=func.__name__)
            return await retry_manager.retry_async(
                func, *args, context=context, **kwargs
            )

        return wrapper

    return decorator


def retry_sync(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
):
    """Decorator for sync functions with retry logic"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retry_manager = RetryManager(
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                strategy=strategy,
            )
            context = ErrorContext(operation=func.__name__)
            return retry_manager.retry_sync(func, *args, context=context, **kwargs)

        return wrapper

    return decorator


def circuit_breaker(
    service_name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    success_threshold: int = 3,
    expected_exception: Type[Exception] = Exception,
):
    """Decorator for functions with circuit breaker protection"""

    def decorator(func: Callable) -> Callable:
        cb = get_circuit_breaker(
            service_name=service_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            expected_exception=expected_exception,
        )

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await cb.call_async(func, *args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return cb.call_sync(func, *args, **kwargs)

            return sync_wrapper

    return decorator


def resilient(
    service_name: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
):
    """Combined decorator for retry logic and circuit breaker"""

    def decorator(func: Callable) -> Callable:
        # Apply circuit breaker first, then retry
        func = circuit_breaker(
            service_name=service_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )(func)

        if asyncio.iscoroutinefunction(func):
            func = retry_async(max_retries=max_retries, base_delay=base_delay)(func)
        else:
            func = retry_sync(max_retries=max_retries, base_delay=base_delay)(func)

        return func

    return decorator
