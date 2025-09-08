"""Input validation utilities for security."""

import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import unquote

from app.security.models import (
    SecurityContext,
    ThreatType,
    ValidationResult,
    ValidationStatus,
)


class InputValidator:
    """Utility class for input validation and sanitization."""

    # Common dangerous patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"('|\"|;|--|\*|\/\*|\*\/)",
        r"(\bUNION\b.*\bSELECT\b)",
        r"(\bINSERT\b.*\bINTO\b)",
        r"(\bDROP\b.*\bTABLE\b)",
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
        r"<link[^>]*>",
        r"<meta[^>]*>",
        r"vbscript:",
        r"data:text/html",
    ]

    COMMAND_INJECTION_PATTERNS = [
        r"(\||&|;|`|\$\(|\${)",
        r"\b(rm|del|format|shutdown|reboot|kill|ps|ls|cat|grep|find|wget|curl)\b",
        r"(\.\.\/|\.\.\\|\/etc\/|\/bin\/|\/usr\/)",
        r"(\$\{.*\}|\$\(.*\))",
        r"(&&|\|\||>>|<<)",
    ]

    PATH_TRAVERSAL_PATTERNS = [
        r"(\.\.\/|\.\.\\)",
        r"(\/etc\/|\/bin\/|\/usr\/|\/var\/|\/tmp\/|\/root\/)",
        r"(\\windows\\|\\system32\\|\\program files\\)",
        r"(\.\.%2f|\.\.%5c)",
        r"(%2e%2e%2f|%2e%2e%5c)",
    ]

    LDAP_INJECTION_PATTERNS = [
        r"(\*|\(|\)|\\|\||&)",
        r"(\x00|\x01|\x02|\x03|\x04|\x05|\x06|\x07)",
    ]

    def __init__(self):
        """Initialize validator with compiled patterns."""
        self.sql_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.SQL_INJECTION_PATTERNS
        ]
        self.xss_patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.XSS_PATTERNS
        ]
        self.cmd_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.COMMAND_INJECTION_PATTERNS
        ]
        self.path_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.PATH_TRAVERSAL_PATTERNS
        ]
        self.ldap_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.LDAP_INJECTION_PATTERNS
        ]

    def validate_string(
        self,
        value: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        allowed_chars: Optional[str] = None,
        blocked_patterns: Optional[List[str]] = None,
    ) -> ValidationResult:
        """Validate a string input."""
        if not isinstance(value, str):
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message="Input must be a string",
            )

        # Length validation
        if max_length and len(value) > max_length:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message=f"Input too long: {len(value)} > {max_length}",
            )

        if min_length and len(value) < min_length:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message=f"Input too short: {len(value)} < {min_length}",
            )

        # Character validation
        if allowed_chars:
            invalid_chars = set(value) - set(allowed_chars)
            if invalid_chars:
                return ValidationResult(
                    status=ValidationStatus.INVALID,
                    message=f"Invalid characters: {', '.join(invalid_chars)}",
                )

        # Pattern validation
        threats = []
        if blocked_patterns:
            for pattern in blocked_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    threats.append(ThreatType.SUSPICIOUS_PATTERN)

        # Security threat detection
        threats.extend(self._detect_threats(value))

        if threats:
            return ValidationResult(
                status=ValidationStatus.SUSPICIOUS,
                sanitized_input=self._sanitize_string(value),
                threats_detected=threats,
                message="Potentially malicious content detected",
            )

        return ValidationResult(
            status=ValidationStatus.VALID,
            sanitized_input=value,
        )

    def validate_email(self, email: str) -> ValidationResult:
        """Validate email address."""
        if not isinstance(email, str):
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message="Email must be a string",
            )

        # Basic email pattern
        email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

        if not email_pattern.match(email):
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message="Invalid email format",
            )

        # Check for suspicious patterns
        threats = self._detect_threats(email)
        if threats:
            return ValidationResult(
                status=ValidationStatus.SUSPICIOUS,
                threats_detected=threats,
                message="Suspicious email content detected",
            )

        return ValidationResult(
            status=ValidationStatus.VALID,
            sanitized_input=email.lower().strip(),
        )

    def validate_url(self, url: str) -> ValidationResult:
        """Validate URL."""
        if not isinstance(url, str):
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message="URL must be a string",
            )

        # Basic URL pattern
        url_pattern = re.compile(
            r"^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$"
        )

        if not url_pattern.match(url):
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message="Invalid URL format",
            )

        # Check for suspicious patterns
        threats = self._detect_threats(url)
        if threats:
            return ValidationResult(
                status=ValidationStatus.SUSPICIOUS,
                threats_detected=threats,
                message="Suspicious URL content detected",
            )

        return ValidationResult(
            status=ValidationStatus.VALID,
            sanitized_input=url.strip(),
        )

    def validate_filename(
        self, filename: str, allowed_extensions: Optional[List[str]] = None
    ) -> ValidationResult:
        """Validate filename."""
        if not isinstance(filename, str):
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message="Filename must be a string",
            )

        # Check for path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                message="Filename contains path traversal characters",
            )

        # Check extension
        if allowed_extensions:
            file_ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
            if file_ext not in allowed_extensions:
                return ValidationResult(
                    status=ValidationStatus.INVALID,
                    message=f"File extension not allowed: {file_ext}",
                )

        # Check for suspicious patterns
        threats = self._detect_threats(filename)
        if threats:
            return ValidationResult(
                status=ValidationStatus.SUSPICIOUS,
                threats_detected=threats,
                message="Suspicious filename content detected",
            )

        return ValidationResult(
            status=ValidationStatus.VALID,
            sanitized_input=filename.strip(),
        )

    def validate_json(self, data: Union[str, Dict, List]) -> ValidationResult:
        """Validate JSON data."""
        import json

        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError as e:
                return ValidationResult(
                    status=ValidationStatus.INVALID,
                    message=f"Invalid JSON: {e}",
                )
        else:
            parsed_data = data

        # Convert back to string for threat detection
        json_str = json.dumps(parsed_data) if not isinstance(data, str) else data

        # Check for suspicious patterns
        threats = self._detect_threats(json_str)
        if threats:
            return ValidationResult(
                status=ValidationStatus.SUSPICIOUS,
                threats_detected=threats,
                message="Suspicious JSON content detected",
            )

        return ValidationResult(
            status=ValidationStatus.VALID,
            sanitized_input=json_str,
        )

    def sanitize_html(self, html: str) -> str:
        """Sanitize HTML content."""
        if not isinstance(html, str):
            return ""

        # Remove script tags
        html = re.sub(
            r"<script[^>]*>.*?</script>", "", html, flags=re.IGNORECASE | re.DOTALL
        )

        # Remove event handlers
        html = re.sub(r"on\w+\s*=\s*[\"'][^\"']*[\"']", "", html, flags=re.IGNORECASE)

        # Remove javascript: links
        html = re.sub(r"javascript:", "", html, flags=re.IGNORECASE)

        # Remove dangerous tags
        dangerous_tags = ["iframe", "object", "embed", "link", "meta", "style"]
        for tag in dangerous_tags:
            html = re.sub(
                f"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.IGNORECASE | re.DOTALL
            )
            html = re.sub(f"<{tag}[^>]*/>", "", html, flags=re.IGNORECASE)

        return html.strip()

    def _detect_threats(self, value: str) -> List[ThreatType]:
        """Detect security threats in input."""
        threats = []
        decoded_value = unquote(value)  # URL decode to catch encoded attacks

        # SQL Injection
        for pattern in self.sql_patterns:
            if pattern.search(decoded_value):
                threats.append(ThreatType.SQL_INJECTION)
                break

        # XSS
        for pattern in self.xss_patterns:
            if pattern.search(decoded_value):
                threats.append(ThreatType.XSS)
                break

        # Command Injection
        for pattern in self.cmd_patterns:
            if pattern.search(decoded_value):
                threats.append(ThreatType.COMMAND_INJECTION)
                break

        # Path Traversal
        for pattern in self.path_patterns:
            if pattern.search(decoded_value):
                threats.append(ThreatType.PATH_TRAVERSAL)
                break

        # LDAP Injection
        for pattern in self.ldap_patterns:
            if pattern.search(decoded_value):
                threats.append(ThreatType.SUSPICIOUS_PATTERN)
                break

        return threats

    def _sanitize_string(self, value: str) -> str:
        """Sanitize string by removing dangerous patterns."""
        sanitized = value

        # Remove SQL injection patterns
        for pattern in self.sql_patterns:
            sanitized = pattern.sub("", sanitized)

        # Remove XSS patterns
        for pattern in self.xss_patterns:
            sanitized = pattern.sub("", sanitized)

        # Remove command injection patterns
        for pattern in self.cmd_patterns:
            sanitized = pattern.sub("", sanitized)

        # Remove path traversal patterns
        for pattern in self.path_patterns:
            sanitized = pattern.sub("", sanitized)

        return sanitized.strip()
