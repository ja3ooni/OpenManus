"""
Configuration validation and schema enforcement for OpenManus.

This module provides comprehensive validation, schema enforcement,
and configuration management capabilities for production environments.
"""

import json
import os
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, ValidationError, validator


class ConfigValidationLevel(Enum):
    """Configuration validation levels"""

    STRICT = "strict"  # All validations must pass
    LENIENT = "lenient"  # Warnings for non-critical issues
    MINIMAL = "minimal"  # Only critical validations


class ConfigValidationSeverity(Enum):
    """Validation issue severity levels"""

    CRITICAL = "critical"  # Prevents system startup
    ERROR = "error"  # Causes functionality issues
    WARNING = "warning"  # Potential issues
    INFO = "info"  # Informational


class ConfigValidationIssue(BaseModel):
    """Represents a configuration validation issue"""

    field_path: str
    severity: ConfigValidationSeverity
    message: str
    current_value: Any = None
    suggested_value: Any = None
    documentation_link: Optional[str] = None


class ConfigSchema(BaseModel):
    """Configuration schema definition"""

    version: str = Field(..., description="Schema version")
    required_fields: Set[str] = Field(default_factory=set)
    optional_fields: Set[str] = Field(default_factory=set)
    field_types: Dict[str, str] = Field(default_factory=dict)
    field_constraints: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    environment_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class ConfigValidator:
    """Comprehensive configuration validator"""

    def __init__(self, schema: Optional[ConfigSchema] = None):
        self.schema = schema or self._load_default_schema()
        self.validation_issues: List[ConfigValidationIssue] = []

    def _load_default_schema(self) -> ConfigSchema:
        """Load default configuration schema"""
        return ConfigSchema(
            version="1.0.0",
            required_fields={
                "llm.default.model",
                "llm.default.api_key",
                "llm.default.api_type",
            },
            optional_fields={
                "sandbox.enabled",
                "browser_config.headless",
                "search_config.max_results",
            },
            field_types={
                "llm.default.max_tokens": "int",
                "llm.default.temperature": "float",
                "sandbox.enabled": "bool",
                "browser_config.timeout": "int",
            },
            field_constraints={
                "llm.default.max_tokens": {"min": 1, "max": 100000},
                "llm.default.temperature": {"min": 0.0, "max": 2.0},
                "browser_config.timeout": {"min": 1, "max": 300},
            },
        )

    def validate_config(
        self,
        config: Dict[str, Any],
        level: ConfigValidationLevel = ConfigValidationLevel.STRICT,
    ) -> List[ConfigValidationIssue]:
        """Validate configuration against schema"""
        self.validation_issues = []

        # Validate required fields
        self._validate_required_fields(config)

        # Validate field types
        self._validate_field_types(config)

        # Validate field constraints
        self._validate_field_constraints(config)

        # Validate environment-specific settings
        self._validate_environment_settings(config)

        # Validate security settings
        self._validate_security_settings(config)

        # Filter issues based on validation level
        if level == ConfigValidationLevel.MINIMAL:
            self.validation_issues = [
                issue
                for issue in self.validation_issues
                if issue.severity == ConfigValidationSeverity.CRITICAL
            ]
        elif level == ConfigValidationLevel.LENIENT:
            self.validation_issues = [
                issue
                for issue in self.validation_issues
                if issue.severity
                in [ConfigValidationSeverity.CRITICAL, ConfigValidationSeverity.ERROR]
            ]

        return self.validation_issues

    def _validate_required_fields(self, config: Dict[str, Any]):
        """Validate that all required fields are present"""
        for field_path in self.schema.required_fields:
            if not self._get_nested_value(config, field_path):
                self.validation_issues.append(
                    ConfigValidationIssue(
                        field_path=field_path,
                        severity=ConfigValidationSeverity.CRITICAL,
                        message=f"Required field '{field_path}' is missing",
                        documentation_link=f"https://docs.openmanus.ai/config#{field_path.replace('.', '-')}",
                    )
                )

    def _validate_field_types(self, config: Dict[str, Any]):
        """Validate field types"""
        for field_path, expected_type in self.schema.field_types.items():
            value = self._get_nested_value(config, field_path)
            if value is not None and not self._check_type(value, expected_type):
                self.validation_issues.append(
                    ConfigValidationIssue(
                        field_path=field_path,
                        severity=ConfigValidationSeverity.ERROR,
                        message=f"Field '{field_path}' should be of type {expected_type}, got {type(value).__name__}",
                        current_value=value,
                        suggested_value=self._suggest_type_conversion(
                            value, expected_type
                        ),
                    )
                )

    def _validate_field_constraints(self, config: Dict[str, Any]):
        """Validate field constraints"""
        for field_path, constraints in self.schema.field_constraints.items():
            value = self._get_nested_value(config, field_path)
            if value is not None:
                self._check_constraints(field_path, value, constraints)

    def _validate_environment_settings(self, config: Dict[str, Any]):
        """Validate environment-specific settings"""
        environment = config.get("environment", "development")

        if environment == "production":
            # Production-specific validations
            self._validate_production_settings(config)
        elif environment == "development":
            # Development-specific validations
            self._validate_development_settings(config)

    def _validate_security_settings(self, config: Dict[str, Any]):
        """Validate security-related settings"""
        # Check for hardcoded secrets
        self._check_for_hardcoded_secrets(config)

        # Validate SSL/TLS settings
        self._validate_ssl_settings(config)

        # Check sandbox settings
        self._validate_sandbox_security(config)

    def _validate_production_settings(self, config: Dict[str, Any]):
        """Validate production-specific settings"""
        production_config = config.get("production", {})

        if not production_config:
            self.validation_issues.append(
                ConfigValidationIssue(
                    field_path="production",
                    severity=ConfigValidationSeverity.CRITICAL,
                    message="Production configuration is required in production environment",
                )
            )
            return

        # Check logging level
        log_level = production_config.get("log_level", "INFO")
        if log_level == "DEBUG":
            self.validation_issues.append(
                ConfigValidationIssue(
                    field_path="production.log_level",
                    severity=ConfigValidationSeverity.WARNING,
                    message="DEBUG logging is not recommended in production",
                    current_value=log_level,
                    suggested_value="INFO",
                )
            )

        # Check metrics configuration
        if not production_config.get("enable_metrics", False):
            self.validation_issues.append(
                ConfigValidationIssue(
                    field_path="production.enable_metrics",
                    severity=ConfigValidationSeverity.WARNING,
                    message="Metrics should be enabled in production for monitoring",
                )
            )

    def _validate_development_settings(self, config: Dict[str, Any]):
        """Validate development-specific settings"""
        # Check for development-specific warnings
        if config.get("sandbox", {}).get("enabled", True) is False:
            self.validation_issues.append(
                ConfigValidationIssue(
                    field_path="sandbox.enabled",
                    severity=ConfigValidationSeverity.WARNING,
                    message="Sandbox should be enabled in development for safety",
                )
            )

    def _check_for_hardcoded_secrets(self, config: Dict[str, Any], path: str = ""):
        """Check for hardcoded secrets in configuration"""
        if isinstance(config, dict):
            for key, value in config.items():
                current_path = f"{path}.{key}" if path else key

                # Check for potential secrets
                if any(
                    secret_key in key.lower()
                    for secret_key in ["key", "secret", "password", "token"]
                ):
                    if (
                        isinstance(value, str)
                        and len(value) > 10
                        and not value.startswith("${")
                    ):
                        self.validation_issues.append(
                            ConfigValidationIssue(
                                field_path=current_path,
                                severity=ConfigValidationSeverity.ERROR,
                                message=f"Potential hardcoded secret in '{current_path}'. Use environment variables instead.",
                                suggested_value="${ENV_VAR_NAME}",
                            )
                        )

                if isinstance(value, (dict, list)):
                    self._check_for_hardcoded_secrets(value, current_path)

        elif isinstance(config, list):
            for i, item in enumerate(config):
                current_path = f"{path}[{i}]"
                if isinstance(item, (dict, list)):
                    self._check_for_hardcoded_secrets(item, current_path)

    def _validate_ssl_settings(self, config: Dict[str, Any]):
        """Validate SSL/TLS settings"""
        # Check LLM API URLs
        llm_config = config.get("llm", {})
        for llm_name, llm_settings in llm_config.items():
            if isinstance(llm_settings, dict):
                base_url = llm_settings.get("base_url", "")
                if base_url and base_url.startswith("http://"):
                    self.validation_issues.append(
                        ConfigValidationIssue(
                            field_path=f"llm.{llm_name}.base_url",
                            severity=ConfigValidationSeverity.WARNING,
                            message="HTTP URLs are not secure. Consider using HTTPS.",
                            current_value=base_url,
                        )
                    )

    def _validate_sandbox_security(self, config: Dict[str, Any]):
        """Validate sandbox security settings"""
        sandbox_config = config.get("sandbox", {})

        if sandbox_config.get("enabled", True) is False:
            environment = config.get("environment", "development")
            if environment == "production":
                self.validation_issues.append(
                    ConfigValidationIssue(
                        field_path="sandbox.enabled",
                        severity=ConfigValidationSeverity.ERROR,
                        message="Sandbox should be enabled in production for security",
                    )
                )

    def _get_nested_value(self, config: Dict[str, Any], field_path: str) -> Any:
        """Get nested value from configuration using dot notation"""
        keys = field_path.split(".")
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        type_map = {
            "str": str,
            "int": int,
            "float": (int, float),  # int is acceptable for float
            "bool": bool,
            "list": list,
            "dict": dict,
        }

        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)

        return True

    def _suggest_type_conversion(self, value: Any, expected_type: str) -> Any:
        """Suggest type conversion for invalid values"""
        try:
            if expected_type == "int":
                return int(value)
            elif expected_type == "float":
                return float(value)
            elif expected_type == "bool":
                if isinstance(value, str):
                    return value.lower() in ["true", "1", "yes", "on"]
                return bool(value)
            elif expected_type == "str":
                return str(value)
        except (ValueError, TypeError):
            pass

        return None

    def _check_constraints(
        self, field_path: str, value: Any, constraints: Dict[str, Any]
    ):
        """Check field constraints"""
        for constraint_type, constraint_value in constraints.items():
            if constraint_type == "min" and value < constraint_value:
                self.validation_issues.append(
                    ConfigValidationIssue(
                        field_path=field_path,
                        severity=ConfigValidationSeverity.ERROR,
                        message=f"Value {value} is below minimum {constraint_value}",
                        current_value=value,
                        suggested_value=constraint_value,
                    )
                )
            elif constraint_type == "max" and value > constraint_value:
                self.validation_issues.append(
                    ConfigValidationIssue(
                        field_path=field_path,
                        severity=ConfigValidationSeverity.ERROR,
                        message=f"Value {value} exceeds maximum {constraint_value}",
                        current_value=value,
                        suggested_value=constraint_value,
                    )
                )
            elif constraint_type == "pattern" and isinstance(value, str):
                if not re.match(constraint_value, value):
                    self.validation_issues.append(
                        ConfigValidationIssue(
                            field_path=field_path,
                            severity=ConfigValidationSeverity.ERROR,
                            message=f"Value '{value}' does not match required pattern '{constraint_value}'",
                        )
                    )

    def generate_validation_report(self) -> str:
        """Generate a human-readable validation report"""
        if not self.validation_issues:
            return "✅ Configuration validation passed with no issues."

        report = ["🔍 Configuration Validation Report", "=" * 40, ""]

        # Group issues by severity
        issues_by_severity = {}
        for issue in self.validation_issues:
            severity = issue.severity.value
            if severity not in issues_by_severity:
                issues_by_severity[severity] = []
            issues_by_severity[severity].append(issue)

        # Report issues by severity
        severity_icons = {"critical": "🚨", "error": "❌", "warning": "⚠️", "info": "ℹ️"}

        for severity in ["critical", "error", "warning", "info"]:
            if severity in issues_by_severity:
                issues = issues_by_severity[severity]
                report.append(
                    f"{severity_icons[severity]} {severity.upper()} ({len(issues)} issues)"
                )
                report.append("-" * 30)

                for issue in issues:
                    report.append(f"Field: {issue.field_path}")
                    report.append(f"Issue: {issue.message}")

                    if issue.current_value is not None:
                        report.append(f"Current: {issue.current_value}")

                    if issue.suggested_value is not None:
                        report.append(f"Suggested: {issue.suggested_value}")

                    if issue.documentation_link:
                        report.append(f"Docs: {issue.documentation_link}")

                    report.append("")

        # Summary
        critical_count = len(issues_by_severity.get("critical", []))
        error_count = len(issues_by_severity.get("error", []))

        if critical_count > 0:
            report.append(
                f"🚨 {critical_count} critical issue(s) must be fixed before startup."
            )
        elif error_count > 0:
            report.append(
                f"❌ {error_count} error(s) should be addressed for optimal operation."
            )
        else:
            report.append(
                "⚠️ Only warnings found. System can start but consider addressing them."
            )

        return "\n".join(report)


class ConfigDocumentationGenerator:
    """Generates configuration documentation"""

    def __init__(self, schema: ConfigSchema):
        self.schema = schema

    def generate_markdown_docs(self) -> str:
        """Generate markdown documentation for configuration"""
        docs = [
            "# OpenManus Configuration Reference",
            "",
            "This document describes all available configuration options for OpenManus.",
            "",
            "## Required Configuration",
            "",
        ]

        # Required fields
        for field in sorted(self.schema.required_fields):
            field_type = self.schema.field_types.get(field, "string")
            constraints = self.schema.field_constraints.get(field, {})

            docs.append(f"### `{field}`")
            docs.append(f"- **Type**: {field_type}")
            docs.append("- **Required**: Yes")

            if constraints:
                docs.append("- **Constraints**:")
                for constraint, value in constraints.items():
                    docs.append(f"  - {constraint}: {value}")

            docs.append("")

        # Optional fields
        docs.extend(["## Optional Configuration", ""])

        for field in sorted(self.schema.optional_fields):
            field_type = self.schema.field_types.get(field, "string")
            constraints = self.schema.field_constraints.get(field, {})

            docs.append(f"### `{field}`")
            docs.append(f"- **Type**: {field_type}")
            docs.append("- **Required**: No")

            if constraints:
                docs.append("- **Constraints**:")
                for constraint, value in constraints.items():
                    docs.append(f"  - {constraint}: {value}")

            docs.append("")

        return "\n".join(docs)

    def generate_json_schema(self) -> Dict[str, Any]:
        """Generate JSON schema for configuration"""
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "OpenManus Configuration Schema",
            "type": "object",
            "properties": {},
            "required": list(self.schema.required_fields),
        }

        # Add field definitions
        all_fields = self.schema.required_fields.union(self.schema.optional_fields)
        for field in all_fields:
            field_type = self.schema.field_types.get(field, "string")
            constraints = self.schema.field_constraints.get(field, {})

            field_schema = {"type": field_type}

            if "min" in constraints:
                field_schema["minimum"] = constraints["min"]
            if "max" in constraints:
                field_schema["maximum"] = constraints["max"]
            if "pattern" in constraints:
                field_schema["pattern"] = constraints["pattern"]

            # Handle nested fields
            field_parts = field.split(".")
            current_schema = schema["properties"]

            for i, part in enumerate(field_parts):
                if i == len(field_parts) - 1:
                    current_schema[part] = field_schema
                else:
                    if part not in current_schema:
                        current_schema[part] = {"type": "object", "properties": {}}
                    current_schema = current_schema[part]["properties"]

        return schema
