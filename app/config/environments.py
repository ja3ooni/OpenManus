"""
Environment-specific configuration management for OpenManus.

This module provides environment-specific configuration overlays,
allowing different settings for development, staging, and production
environments while maintaining a single base configuration.
"""

import os
import tomllib
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from app.logger import logger


class Environment(Enum):
    """Supported environments"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class EnvironmentConfig(BaseModel):
    """Environment-specific configuration"""

    name: Environment
    description: str = ""
    inherits_from: Optional[Environment] = None
    config_overrides: Dict[str, Any] = Field(default_factory=dict)
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    required_env_vars: List[str] = Field(default_factory=list)
    feature_flags: Dict[str, bool] = Field(default_factory=dict)
    resource_limits: Dict[str, Any] = Field(default_factory=dict)


class ConfigEnvironmentManager:
    """Manages environment-specific configuration overlays"""

    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)
        self.environments_dir = self.config_dir / "environments"
        self.environments_dir.mkdir(exist_ok=True)

        self.current_environment = self._detect_current_environment()
        self.environment_configs: Dict[Environment, EnvironmentConfig] = {}

        # Load environment configurations
        self._load_environment_configs()

        # Ensure default environments exist
        self._ensure_default_environments()

    def _detect_current_environment(self) -> Environment:
        """Detect current environment from environment variables"""
        env_name = os.getenv("OPENMANUS_ENV", os.getenv("ENV", "development")).lower()

        try:
            return Environment(env_name)
        except ValueError:
            logger.warning(
                f"Unknown environment '{env_name}', defaulting to development"
            )
            return Environment.DEVELOPMENT

    def _load_environment_configs(self):
        """Load all environment configuration files"""
        for env_file in self.environments_dir.glob("*.toml"):
            try:
                env_name = env_file.stem
                environment = Environment(env_name)

                with open(env_file, "rb") as f:
                    config_data = tomllib.load(f)

                env_config = EnvironmentConfig(name=environment, **config_data)

                self.environment_configs[environment] = env_config
                logger.debug(f"Loaded environment config for {env_name}")

            except Exception as e:
                logger.error(f"Failed to load environment config {env_file}: {e}")

    def _ensure_default_environments(self):
        """Ensure default environment configurations exist"""
        default_configs = {
            Environment.DEVELOPMENT: {
                "description": "Development environment with debug features enabled",
                "config_overrides": {
                    "llm": {"default": {"temperature": 0.1, "max_tokens": 4096}},
                    "sandbox": {"enabled": True, "timeout": 30},
                    "browser_config": {"headless": False, "timeout": 30},
                    "production": {
                        "log_level": "DEBUG",
                        "enable_metrics": False,
                        "enable_profiling": True,
                    },
                },
                "feature_flags": {
                    "debug_mode": True,
                    "experimental_features": True,
                    "detailed_logging": True,
                },
                "resource_limits": {
                    "max_memory_mb": 2048,
                    "max_concurrent_requests": 5,
                },
            },
            Environment.TESTING: {
                "description": "Testing environment for automated tests",
                "inherits_from": Environment.DEVELOPMENT,
                "config_overrides": {
                    "llm": {
                        "default": {
                            "temperature": 0.0,  # Deterministic for testing
                            "max_tokens": 1024,
                        }
                    },
                    "sandbox": {"enabled": True, "timeout": 10},
                    "browser_config": {"headless": True, "timeout": 10},
                    "production": {"log_level": "WARNING", "enable_metrics": False},
                },
                "feature_flags": {
                    "debug_mode": False,
                    "experimental_features": False,
                    "mock_external_apis": True,
                },
                "resource_limits": {
                    "max_memory_mb": 1024,
                    "max_concurrent_requests": 2,
                },
            },
            Environment.STAGING: {
                "description": "Staging environment for pre-production testing",
                "config_overrides": {
                    "llm": {"default": {"temperature": 0.0, "max_tokens": 8192}},
                    "sandbox": {"enabled": True, "timeout": 60},
                    "browser_config": {"headless": True, "timeout": 60},
                    "production": {
                        "log_level": "INFO",
                        "enable_metrics": True,
                        "enable_health_checks": True,
                    },
                },
                "required_env_vars": ["OPENMANUS_API_KEY", "OPENMANUS_DATABASE_URL"],
                "feature_flags": {
                    "debug_mode": False,
                    "experimental_features": False,
                    "performance_monitoring": True,
                },
                "resource_limits": {
                    "max_memory_mb": 4096,
                    "max_concurrent_requests": 20,
                },
            },
            Environment.PRODUCTION: {
                "description": "Production environment with optimized settings",
                "config_overrides": {
                    "llm": {"default": {"temperature": 0.1, "max_tokens": 8192}},
                    "sandbox": {"enabled": True, "timeout": 120},
                    "browser_config": {"headless": True, "timeout": 120},
                    "production": {
                        "log_level": "INFO",
                        "enable_metrics": True,
                        "enable_health_checks": True,
                        "enable_profiling": False,
                        "max_concurrent_requests": 50,
                        "request_timeout": 300,
                    },
                },
                "required_env_vars": [
                    "OPENMANUS_API_KEY",
                    "OPENMANUS_DATABASE_URL",
                    "OPENMANUS_SECRET_KEY",
                ],
                "feature_flags": {
                    "debug_mode": False,
                    "experimental_features": False,
                    "performance_monitoring": True,
                    "security_hardening": True,
                },
                "resource_limits": {
                    "max_memory_mb": 8192,
                    "max_concurrent_requests": 100,
                },
            },
        }

        for environment, config_data in default_configs.items():
            if environment not in self.environment_configs:
                env_config = EnvironmentConfig(name=environment, **config_data)
                self.environment_configs[environment] = env_config
                self._save_environment_config(env_config)

    def _save_environment_config(self, env_config: EnvironmentConfig):
        """Save environment configuration to file"""
        env_file = self.environments_dir / f"{env_config.name.value}.toml"

        try:
            import toml

            config_dict = env_config.dict(exclude={"name"})

            # Convert enum values to strings
            if config_dict.get("inherits_from"):
                config_dict["inherits_from"] = config_dict["inherits_from"].value

            with open(env_file, "w") as f:
                toml.dump(config_dict, f)

            logger.debug(f"Saved environment config for {env_config.name.value}")

        except Exception as e:
            logger.error(
                f"Failed to save environment config {env_config.name.value}: {e}"
            )

    def get_current_environment(self) -> Environment:
        """Get current environment"""
        return self.current_environment

    def set_current_environment(self, environment: Environment):
        """Set current environment"""
        self.current_environment = environment
        logger.info(f"Switched to environment: {environment.value}")

    def get_environment_config(
        self, environment: Environment
    ) -> Optional[EnvironmentConfig]:
        """Get configuration for specific environment"""
        return self.environment_configs.get(environment)

    def apply_environment_overrides(
        self, base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply environment-specific overrides to base configuration"""
        env_config = self.get_environment_config(self.current_environment)
        if not env_config:
            logger.warning(
                f"No configuration found for environment {self.current_environment.value}"
            )
            return base_config

        # Start with base config
        result_config = base_config.copy()

        # Apply inheritance chain
        inheritance_chain = self._build_inheritance_chain(env_config)

        for inherited_env_config in inheritance_chain:
            result_config = self._merge_config_overrides(
                result_config, inherited_env_config.config_overrides
            )

        # Apply environment variables
        result_config = self._apply_environment_variables(result_config, env_config)

        # Validate required environment variables
        self._validate_required_env_vars(env_config)

        # Add environment metadata
        result_config["environment"] = self.current_environment.value
        result_config["environment_config"] = {
            "description": env_config.description,
            "feature_flags": env_config.feature_flags,
            "resource_limits": env_config.resource_limits,
        }

        logger.info(
            f"Applied {self.current_environment.value} environment configuration"
        )
        return result_config

    def _build_inheritance_chain(
        self, env_config: EnvironmentConfig
    ) -> List[EnvironmentConfig]:
        """Build inheritance chain for environment configuration"""
        chain = []
        current_config = env_config
        visited = set()

        while current_config:
            if current_config.name in visited:
                logger.error(
                    f"Circular inheritance detected in environment {current_config.name.value}"
                )
                break

            visited.add(current_config.name)
            chain.insert(0, current_config)  # Insert at beginning for correct order

            if current_config.inherits_from:
                current_config = self.environment_configs.get(
                    current_config.inherits_from
                )
            else:
                current_config = None

        return chain

    def _merge_config_overrides(
        self, base_config: Dict[str, Any], overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge configuration overrides into base configuration"""
        result = base_config.copy()

        for key, value in overrides.items():
            if (
                isinstance(value, dict)
                and key in result
                and isinstance(result[key], dict)
            ):
                # Recursively merge nested dictionaries
                result[key] = self._merge_config_overrides(result[key], value)
            else:
                # Override value
                result[key] = value

        return result

    def _apply_environment_variables(
        self, config: Dict[str, Any], env_config: EnvironmentConfig
    ) -> Dict[str, Any]:
        """Apply environment variable substitutions"""
        result = config.copy()

        # Apply configured environment variables
        for env_var, config_path in env_config.environment_variables.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_config_value(result, config_path, env_value)

        # Apply standard environment variable patterns
        result = self._substitute_env_vars_in_config(result)

        return result

    def _substitute_env_vars_in_config(self, config: Any) -> Any:
        """Recursively substitute environment variables in configuration"""
        if isinstance(config, dict):
            return {
                key: self._substitute_env_vars_in_config(value)
                for key, value in config.items()
            }
        elif isinstance(config, list):
            return [self._substitute_env_vars_in_config(item) for item in config]
        elif (
            isinstance(config, str) and config.startswith("${") and config.endswith("}")
        ):
            # Environment variable substitution
            env_var = config[2:-1]  # Remove ${ and }
            default_value = None

            # Handle default values: ${VAR:default}
            if ":" in env_var:
                env_var, default_value = env_var.split(":", 1)

            env_value = os.getenv(env_var, default_value)
            if env_value is None:
                logger.warning(
                    f"Environment variable {env_var} not found and no default provided"
                )
                return config

            # Try to convert to appropriate type
            return self._convert_env_var_type(env_value)
        else:
            return config

    def _convert_env_var_type(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type"""
        # Boolean conversion
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Integer conversion
        try:
            if "." not in value:
                return int(value)
        except ValueError:
            pass

        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _set_nested_config_value(self, config: Dict[str, Any], path: str, value: Any):
        """Set nested configuration value using dot notation"""
        keys = path.split(".")
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _validate_required_env_vars(self, env_config: EnvironmentConfig):
        """Validate that required environment variables are set"""
        missing_vars = []

        for env_var in env_config.required_env_vars:
            if not os.getenv(env_var):
                missing_vars.append(env_var)

        if missing_vars:
            error_msg = f"Missing required environment variables for {env_config.name.value}: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def get_feature_flag(self, flag_name: str, default: bool = False) -> bool:
        """Get feature flag value for current environment"""
        env_config = self.get_environment_config(self.current_environment)
        if env_config:
            return env_config.feature_flags.get(flag_name, default)
        return default

    def get_resource_limit(self, limit_name: str, default: Any = None) -> Any:
        """Get resource limit for current environment"""
        env_config = self.get_environment_config(self.current_environment)
        if env_config:
            return env_config.resource_limits.get(limit_name, default)
        return default

    def list_environments(self) -> List[Environment]:
        """List all available environments"""
        return list(self.environment_configs.keys())

    def create_environment(
        self,
        name: str,
        description: str = "",
        inherits_from: Optional[Environment] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> EnvironmentConfig:
        """Create a new environment configuration"""
        try:
            environment = Environment(name)
        except ValueError:
            raise ValueError(f"Invalid environment name: {name}")

        if environment in self.environment_configs:
            raise ValueError(f"Environment {name} already exists")

        env_config = EnvironmentConfig(
            name=environment,
            description=description,
            inherits_from=inherits_from,
            config_overrides=config_overrides or {},
        )

        self.environment_configs[environment] = env_config
        self._save_environment_config(env_config)

        logger.info(f"Created new environment: {name}")
        return env_config

    def delete_environment(self, environment: Environment) -> bool:
        """Delete an environment configuration"""
        if environment not in self.environment_configs:
            return False

        if environment == self.current_environment:
            raise ValueError("Cannot delete current environment")

        # Check if any other environments inherit from this one
        dependent_envs = [
            env_config.name
            for env_config in self.environment_configs.values()
            if env_config.inherits_from == environment
        ]

        if dependent_envs:
            raise ValueError(
                f"Cannot delete environment {environment.value}, it is inherited by: {[e.value for e in dependent_envs]}"
            )

        # Remove configuration file
        env_file = self.environments_dir / f"{environment.value}.toml"
        if env_file.exists():
            env_file.unlink()

        # Remove from memory
        del self.environment_configs[environment]

        logger.info(f"Deleted environment: {environment.value}")
        return True

    def get_environment_summary(self) -> Dict[str, Any]:
        """Get summary of environment configuration"""
        current_env_config = self.get_environment_config(self.current_environment)

        return {
            "current_environment": self.current_environment.value,
            "available_environments": [env.value for env in self.list_environments()],
            "current_config": {
                "description": (
                    current_env_config.description if current_env_config else ""
                ),
                "inherits_from": (
                    current_env_config.inherits_from.value
                    if current_env_config and current_env_config.inherits_from
                    else None
                ),
                "feature_flags": (
                    current_env_config.feature_flags if current_env_config else {}
                ),
                "resource_limits": (
                    current_env_config.resource_limits if current_env_config else {}
                ),
            },
            "required_env_vars": (
                current_env_config.required_env_vars if current_env_config else []
            ),
            "missing_env_vars": [
                var
                for var in (
                    current_env_config.required_env_vars if current_env_config else []
                )
                if not os.getenv(var)
            ],
        }
