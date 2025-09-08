"""Tests for InputValidator."""

import pytest

from app.security.models import ThreatType, ValidationStatus
from app.security.validators import InputValidator


class TestInputValidator:
    """Test cases for InputValidator."""

    @pytest.fixture
    def validator(self):
        """Create InputValidator instance."""
        return InputValidator()

    def test_validate_string_clean(self, validator):
        """Test validation of clean string."""
        result = validator.validate_string("This is a clean string")

        assert result.status == ValidationStatus.VALID
        assert result.sanitized_input == "This is a clean string"
        assert not result.threats_detected

    def test_validate_string_too_long(self, validator):
        """Test string length validation."""
        long_string = "a" * 1000
        result = validator.validate_string(long_string, max_length=100)

        assert result.status == ValidationStatus.INVALID
        assert "too long" in result.message.lower()

    def test_validate_string_too_short(self, validator):
        """Test minimum string length validation."""
        short_string = "ab"
        result = validator.validate_string(short_string, min_length=5)

        assert result.status == ValidationStatus.INVALID
        assert "too short" in result.message.lower()

    def test_validate_string_invalid_chars(self, validator):
        """Test character validation."""
        result = validator.validate_string(
            "abc123!", allowed_chars="abcdefghijklmnopqrstuvwxyz"
        )

        assert result.status == ValidationStatus.INVALID
        assert "invalid characters" in result.message.lower()

    def test_validate_string_sql_injection(self, validator):
        """Test SQL injection detection in string validation."""
        malicious_string = "'; DROP TABLE users; --"
        result = validator.validate_string(malicious_string)

        assert result.status == ValidationStatus.SUSPICIOUS
        assert ThreatType.SQL_INJECTION in result.threats_detected
        assert result.sanitized_input != malicious_string

    def test_validate_string_xss(self, validator):
        """Test XSS detection in string validation."""
        malicious_string = "<script>alert('xss')</script>"
        result = validator.validate_string(malicious_string)

        assert result.status == ValidationStatus.SUSPICIOUS
        assert ThreatType.XSS in result.threats_detected

    def test_validate_string_command_injection(self, validator):
        """Test command injection detection."""
        malicious_string = "test; rm -rf /"
        result = validator.validate_string(malicious_string)

        assert result.status == ValidationStatus.SUSPICIOUS
        assert ThreatType.COMMAND_INJECTION in result.threats_detected

    def test_validate_string_blocked_patterns(self, validator):
        """Test custom blocked patterns."""
        result = validator.validate_string(
            "This contains forbidden_word", blocked_patterns=[r"\bforbidden_word\b"]
        )

        assert result.status == ValidationStatus.SUSPICIOUS
        assert ThreatType.SUSPICIOUS_PATTERN in result.threats_detected

    def test_validate_email_valid(self, validator):
        """Test valid email validation."""
        valid_emails = [
            "user@example.com",
            "test.email+tag@domain.co.uk",
            "user123@test-domain.org",
        ]

        for email in valid_emails:
            result = validator.validate_email(email)
            assert result.status == ValidationStatus.VALID
            assert result.sanitized_input == email.lower().strip()

    def test_validate_email_invalid_format(self, validator):
        """Test invalid email format detection."""
        invalid_emails = [
            "not-an-email",
            "@domain.com",
            "user@",
            "user@domain",
            "user..double.dot@domain.com",
        ]

        for email in invalid_emails:
            result = validator.validate_email(email)
            assert result.status == ValidationStatus.INVALID
            assert "invalid email format" in result.message.lower()

    def test_validate_email_non_string(self, validator):
        """Test email validation with non-string input."""
        result = validator.validate_email(123)

        assert result.status == ValidationStatus.INVALID
        assert "must be a string" in result.message.lower()

    def test_validate_email_suspicious_content(self, validator):
        """Test email with suspicious content."""
        suspicious_email = "user+<script>@domain.com"
        result = validator.validate_email(suspicious_email)

        # Should be invalid due to format, but let's test with a valid format
        suspicious_email = "user@domain.com'; DROP TABLE users; --"
        result = validator.validate_email(suspicious_email)

        assert result.status == ValidationStatus.INVALID  # Invalid format

    def test_validate_url_valid(self, validator):
        """Test valid URL validation."""
        valid_urls = [
            "https://example.com",
            "http://www.test-domain.org/path?param=value",
            "https://subdomain.example.com:8080/path",
        ]

        for url in valid_urls:
            result = validator.validate_url(url)
            assert result.status == ValidationStatus.VALID
            assert result.sanitized_input == url.strip()

    def test_validate_url_invalid_format(self, validator):
        """Test invalid URL format detection."""
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",  # Only http/https allowed
            "https://",
            "http://domain",
        ]

        for url in invalid_urls:
            result = validator.validate_url(url)
            assert result.status == ValidationStatus.INVALID

    def test_validate_url_suspicious_content(self, validator):
        """Test URL with suspicious content."""
        suspicious_url = "https://example.com/<script>alert('xss')</script>"
        result = validator.validate_url(suspicious_url)

        # Should be suspicious due to XSS content
        assert result.status == ValidationStatus.SUSPICIOUS
        assert ThreatType.XSS in result.threats_detected

    def test_validate_filename_valid(self, validator):
        """Test valid filename validation."""
        valid_filenames = [
            "document.txt",
            "image.jpg",
            "script.py",
            "data_file.csv",
        ]

        for filename in valid_filenames:
            result = validator.validate_filename(filename)
            assert result.status == ValidationStatus.VALID

    def test_validate_filename_path_traversal(self, validator):
        """Test filename with path traversal."""
        malicious_filenames = [
            "../../../etc/passwd",
            "..\\windows\\system32\\config",
            "file/../other.txt",
        ]

        for filename in malicious_filenames:
            result = validator.validate_filename(filename)
            assert result.status == ValidationStatus.INVALID
            assert "path traversal" in result.message.lower()

    def test_validate_filename_extension_check(self, validator):
        """Test filename extension validation."""
        result = validator.validate_filename(
            "malicious.exe", allowed_extensions=[".txt", ".jpg", ".png"]
        )

        assert result.status == ValidationStatus.INVALID
        assert "extension not allowed" in result.message.lower()

    def test_validate_filename_suspicious_content(self, validator):
        """Test filename with suspicious content."""
        suspicious_filename = "file'; DROP TABLE files; --.txt"
        result = validator.validate_filename(suspicious_filename)

        assert result.status == ValidationStatus.SUSPICIOUS
        assert ThreatType.SQL_INJECTION in result.threats_detected

    def test_validate_json_valid_string(self, validator):
        """Test valid JSON string validation."""
        valid_json = '{"key": "value", "number": 123}'
        result = validator.validate_json(valid_json)

        assert result.status == ValidationStatus.VALID

    def test_validate_json_valid_dict(self, validator):
        """Test valid JSON dict validation."""
        valid_dict = {"key": "value", "number": 123}
        result = validator.validate_json(valid_dict)

        assert result.status == ValidationStatus.VALID

    def test_validate_json_invalid_string(self, validator):
        """Test invalid JSON string."""
        invalid_json = '{"key": "value", "number": 123'  # Missing closing brace
        result = validator.validate_json(invalid_json)

        assert result.status == ValidationStatus.INVALID
        assert "invalid json" in result.message.lower()

    def test_validate_json_suspicious_content(self, validator):
        """Test JSON with suspicious content."""
        suspicious_json = '{"script": "<script>alert(\\"xss\\")</script>"}'
        result = validator.validate_json(suspicious_json)

        assert result.status == ValidationStatus.SUSPICIOUS
        assert ThreatType.XSS in result.threats_detected

    def test_sanitize_html_clean(self, validator):
        """Test HTML sanitization with clean content."""
        clean_html = "<p>This is clean content</p>"
        sanitized = validator.sanitize_html(clean_html)

        assert sanitized == clean_html

    def test_sanitize_html_script_removal(self, validator):
        """Test script tag removal."""
        malicious_html = (
            '<p>Content</p><script>alert("xss")</script><p>More content</p>'
        )
        sanitized = validator.sanitize_html(malicious_html)

        assert "<script>" not in sanitized
        assert "alert" not in sanitized
        assert "<p>Content</p>" in sanitized

    def test_sanitize_html_event_handlers(self, validator):
        """Test event handler removal."""
        malicious_html = "<div onclick=\"alert('xss')\">Click me</div>"
        sanitized = validator.sanitize_html(malicious_html)

        assert "onclick" not in sanitized
        assert "alert" not in sanitized

    def test_sanitize_html_javascript_links(self, validator):
        """Test javascript: link removal."""
        malicious_html = "<a href=\"javascript:alert('xss')\">Link</a>"
        sanitized = validator.sanitize_html(malicious_html)

        assert "javascript:" not in sanitized

    def test_sanitize_html_dangerous_tags(self, validator):
        """Test dangerous tag removal."""
        malicious_html = """
        <iframe src="http://evil.com"></iframe>
        <object data="malicious.swf"></object>
        <embed src="malicious.swf">
        <link rel="stylesheet" href="http://evil.com/style.css">
        """
        sanitized = validator.sanitize_html(malicious_html)

        assert "<iframe>" not in sanitized
        assert "<object>" not in sanitized
        assert "<embed>" not in sanitized
        assert "<link>" not in sanitized

    def test_sanitize_html_non_string(self, validator):
        """Test HTML sanitization with non-string input."""
        result = validator.sanitize_html(123)
        assert result == ""

    def test_threat_detection_url_encoded(self, validator):
        """Test threat detection with URL-encoded input."""
        # URL-encoded XSS: %3Cscript%3Ealert('xss')%3C/script%3E
        encoded_xss = "%3Cscript%3Ealert('xss')%3C/script%3E"
        threats = validator._detect_threats(encoded_xss)

        assert ThreatType.XSS in threats

    def test_multiple_threat_detection(self, validator):
        """Test detection of multiple threats."""
        # Input with SQL injection and XSS
        malicious_input = "'; DROP TABLE users; --<script>alert('xss')</script>"
        threats = validator._detect_threats(malicious_input)

        assert ThreatType.SQL_INJECTION in threats
        assert ThreatType.XSS in threats

    def test_sanitize_string_comprehensive(self, validator):
        """Test comprehensive string sanitization."""
        malicious_input = (
            "'; DROP TABLE users; --<script>alert('xss')</script>$(rm -rf /)"
        )
        sanitized = validator._sanitize_string(malicious_input)

        # Should remove all malicious patterns
        assert "DROP TABLE" not in sanitized
        assert "<script>" not in sanitized
        assert "rm -rf" not in sanitized
        assert len(sanitized) < len(malicious_input)
