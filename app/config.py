import hashlib
import json
import os
import threading
import time
import tomllib
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, model_validator, validator
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"


class Environment(str, Enum):
    """Supported deployment environments"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Supported log levels"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ConfigVersion(BaseModel):
    """Configuration version tracking"""

    version: str = Field(..., description="Configuration version")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Version timestamp"
    )
    checksum: str = Field(..., description="Configuration checksum")
    environment: Environment = Field(..., description="Target environment")
    description: Optional[str] = Field(None, description="Version description")


class ValidationError(Exception):
    """Configuration validation error"""

    pass


class ConfigurationChangeHandler(FileSystemEventHandler):
    """File system event handler for configuration changes"""

    def __init__(self, config_instance):
        self.config_instance = config_instance
        self.last_modified = {}

    def on_modified(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path.suffix in [".toml", ".json"]:
            # Debounce rapid file changes
            current_time = time.time()
            if file_path in self.last_modified:
                if current_time - self.last_modified[file_path] < 1.0:
                    return

            self.last_modified[file_path] = current_time
            self.config_instance._handle_config_change(file_path)


class LLMSettings(BaseModel):
    model: str = Field(..., description="Model name")
    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(..., description="API key")
    max_tokens: int = Field(4096, description="Maximum number of tokens per request")
    max_input_tokens: Optional[int] = Field(
        None,
        description="Maximum input tokens to use across all requests (None for unlimited)",
    )
    temperature: float = Field(1.0, description="Sampling temperature")
    api_type: str = Field(..., description="Azure, Openai, or Ollama")
    api_version: str = Field(..., description="Azure Openai version if AzureOpenai")

    @validator("max_tokens")
    def validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        if v > 200000:  # Reasonable upper limit
            raise ValueError("max_tokens exceeds reasonable limit (200000)")
        return v

    @validator("temperature")
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @validator("api_type")
    def validate_api_type(cls, v):
        valid_types = ["azure", "openai", "ollama", "aws"]
        if v.lower() not in valid_types:
            raise ValueError(f"api_type must be one of: {valid_types}")
        return v.lower()


class ProxySettings(BaseModel):
    server: str = Field(None, description="Proxy server address")
    username: Optional[str] = Field(None, description="Proxy username")
    password: Optional[str] = Field(None, description="Proxy password")


class SearchSettings(BaseModel):
    engine: str = Field(default="Google", description="Search engine the llm to use")
    fallback_engines: List[str] = Field(
        default_factory=lambda: ["DuckDuckGo", "Baidu", "Bing"],
        description="Fallback search engines to try if the primary engine fails",
    )
    retry_delay: int = Field(
        default=60,
        description="Seconds to wait before retrying all engines again after they all fail",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of times to retry all engines when all fail",
    )
    lang: str = Field(
        default="en",
        description="Language code for search results (e.g., en, zh, fr)",
    )
    country: str = Field(
        default="us",
        description="Country code for search results (e.g., us, cn, uk)",
    )


class RunflowSettings(BaseModel):
    use_data_analysis_agent: bool = Field(
        default=False, description="Enable data analysis agent in run flow"
    )


class BrowserSettings(BaseModel):
    headless: bool = Field(False, description="Whether to run browser in headless mode")
    disable_security: bool = Field(
        True, description="Disable browser security features"
    )
    extra_chromium_args: List[str] = Field(
        default_factory=list, description="Extra arguments to pass to the browser"
    )
    chrome_instance_path: Optional[str] = Field(
        None, description="Path to a Chrome instance to use"
    )
    wss_url: Optional[str] = Field(
        None, description="Connect to a browser instance via WebSocket"
    )
    cdp_url: Optional[str] = Field(
        None, description="Connect to a browser instance via CDP"
    )
    proxy: Optional[ProxySettings] = Field(
        None, description="Proxy settings for the browser"
    )
    max_content_length: int = Field(
        2000, description="Maximum length for content retrieval operations"
    )


class SandboxSettings(BaseModel):
    """Configuration for the execution sandbox"""

    use_sandbox: bool = Field(False, description="Whether to use the sandbox")
    image: str = Field("python:3.12-slim", description="Base image")
    work_dir: str = Field("/workspace", description="Container working directory")
    memory_limit: str = Field("512m", description="Memory limit")
    cpu_limit: float = Field(1.0, description="CPU limit")
    timeout: int = Field(300, description="Default command timeout (seconds)")
    network_enabled: bool = Field(
        False, description="Whether network access is allowed"
    )

    @validator("memory_limit")
    def validate_memory_limit(cls, v):
        import re

        if not re.match(r"^\d+[kmg]?$", v.lower()):
            raise ValueError(
                'memory_limit must be in format like "512m", "1g", "1024k"'
            )
        return v

    @validator("cpu_limit")
    def validate_cpu_limit(cls, v):
        if v <= 0 or v > 32:  # Reasonable CPU limit
            raise ValueError("cpu_limit must be between 0 and 32")
        return v

    @validator("timeout")
    def validate_timeout(cls, v):
        if v <= 0 or v > 3600:  # Max 1 hour
            raise ValueError("timeout must be between 1 and 3600 seconds")
        return v


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server"""

    type: str = Field(..., description="Server connection type (sse or stdio)")
    url: Optional[str] = Field(None, description="Server URL for SSE connections")
    command: Optional[str] = Field(None, description="Command for stdio connections")
    args: List[str] = Field(
        default_factory=list, description="Arguments for stdio command"
    )


class MCPSettings(BaseModel):
    """Configuration for MCP (Model Context Protocol)"""

    server_reference: str = Field(
        "app.mcp.server", description="Module reference for the MCP server"
    )
    servers: Dict[str, MCPServerConfig] = Field(
        default_factory=dict, description="MCP server configurations"
    )

    @classmethod
    def load_server_config(cls) -> Dict[str, MCPServerConfig]:
        """Load MCP server configuration from JSON file"""
        config_path = PROJECT_ROOT / "config" / "mcp.json"

        try:
            config_file = config_path if config_path.exists() else None
            if not config_file:
                return {}

            with config_file.open() as f:
                data = json.load(f)
                servers = {}

                for server_id, server_config in data.get("mcpServers", {}).items():
                    servers[server_id] = MCPServerConfig(
                        type=server_config["type"],
                        url=server_config.get("url"),
                        command=server_config.get("command"),
                        args=server_config.get("args", []),
                    )
                return servers
        except Exception as e:
            raise ValueError(f"Failed to load MCP server config: {e}")


class ProductionSettings(BaseModel):
    """Production-specific configuration settings"""

    environment: Environment = Field(
        Environment.PRODUCTION, description="Deployment environment"
    )
    debug_mode: bool = Field(False, description="Enable debug mode")
    log_level: LogLevel = Field(LogLevel.INFO, description="Logging level")

    # Performance settings
    max_concurrent_requests: int = Field(10, description="Maximum concurrent requests")
    request_timeout: int = Field(300, description="Request timeout in seconds")
    memory_limit: str = Field("2GB", description="Application memory limit")

    # Security settings
    enable_sandbox: bool = Field(True, description="Enable sandbox execution")
    sandbox_timeout: int = Field(300, description="Sandbox timeout in seconds")
    max_file_size: str = Field("100MB", description="Maximum file size")
    allowed_domains: List[str] = Field(
        default_factory=list, description="Allowed domains for external requests"
    )

    # Monitoring settings
    enable_metrics: bool = Field(True, description="Enable metrics collection")
    metrics_port: int = Field(9090, description="Metrics server port")
    health_check_interval: int = Field(
        30, description="Health check interval in seconds"
    )

    # Research settings
    max_research_sources: int = Field(10, description="Maximum research sources")
    research_timeout: int = Field(120, description="Research timeout in seconds")
    enable_cross_referencing: bool = Field(True, description="Enable cross-referencing")

    # Writing settings
    default_writing_style: str = Field(
        "professional", description="Default writing style"
    )
    enable_grammar_check: bool = Field(True, description="Enable grammar checking")
    citation_style: str = Field("APA", description="Default citation style")

    @validator("max_concurrent_requests")
    def validate_concurrent_requests(cls, v):
        if v <= 0 or v > 100:
            raise ValueError("max_concurrent_requests must be between 1 and 100")
        return v

    @validator("request_timeout")
    def validate_request_timeout(cls, v):
        if v <= 0 or v > 3600:
            raise ValueError("request_timeout must be between 1 and 3600 seconds")
        return v

    @validator("metrics_port")
    def validate_metrics_port(cls, v):
        if v < 1024 or v > 65535:
            raise ValueError("metrics_port must be between 1024 and 65535")
        return v


class AppConfig(BaseModel):
    llm: Dict[str, LLMSettings]
    sandbox: Optional[SandboxSettings] = Field(
        None, description="Sandbox configuration"
    )
    browser_config: Optional[BrowserSettings] = Field(
        None, description="Browser configuration"
    )
    search_config: Optional[SearchSettings] = Field(
        None, description="Search configuration"
    )
    mcp_config: Optional[MCPSettings] = Field(None, description="MCP configuration")
    run_flow_config: Optional[RunflowSettings] = Field(
        None, description="Run flow configuration"
    )
    production: Optional[ProductionSettings] = Field(
        None, description="Production configuration"
    )

    # Configuration metadata
    version: Optional[str] = Field(None, description="Configuration version")
    environment: Environment = Field(
        Environment.DEVELOPMENT, description="Current environment"
    )
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="before")
    @classmethod
    def validate_environment_consistency(cls, values):
        """Ensure environment settings are consistent"""
        if isinstance(values, dict):
            env = values.get("environment", Environment.DEVELOPMENT)
            production = values.get("production")

            if env == Environment.PRODUCTION and not production:
                raise ValueError("Production environment requires production settings")

            if production and production.environment != env:
                raise ValueError(
                    "Environment mismatch between main config and production settings"
                )

        return values


class Config:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config = None
                    self._config_versions = []
                    self._change_callbacks = []
                    self._file_observer = None
                    self._hot_reload_enabled = False
                    self._validation_enabled = True
                    self._rollback_enabled = True
                    self._load_initial_config()
                    self._setup_hot_reload()
                    self._initialized = True

    @staticmethod
    def _get_config_path() -> Path:
        root = PROJECT_ROOT
        config_path = root / "config" / "config.toml"
        if config_path.exists():
            return config_path
        example_path = root / "config" / "config.example.toml"
        if example_path.exists():
            return example_path
        raise FileNotFoundError("No configuration file found in config directory")

    def _load_config(self) -> dict:
        config_path = self._get_config_path()
        with config_path.open("rb") as f:
            return tomllib.load(f)

    def _load_initial_config(self):
        raw_config = self._load_config()
        base_llm = raw_config.get("llm", {})
        llm_overrides = {
            k: v for k, v in raw_config.get("llm", {}).items() if isinstance(v, dict)
        }

        default_settings = {
            "model": base_llm.get("model"),
            "base_url": base_llm.get("base_url"),
            "api_key": base_llm.get("api_key"),
            "max_tokens": base_llm.get("max_tokens", 4096),
            "max_input_tokens": base_llm.get("max_input_tokens"),
            "temperature": base_llm.get("temperature", 1.0),
            "api_type": base_llm.get("api_type", ""),
            "api_version": base_llm.get("api_version", ""),
        }

        # handle browser config.
        browser_config = raw_config.get("browser", {})
        browser_settings = None

        if browser_config:
            # handle proxy settings.
            proxy_config = browser_config.get("proxy", {})
            proxy_settings = None

            if proxy_config and proxy_config.get("server"):
                proxy_settings = ProxySettings(
                    **{
                        k: v
                        for k, v in proxy_config.items()
                        if k in ["server", "username", "password"] and v
                    }
                )

            # filter valid browser config parameters.
            valid_browser_params = {
                k: v
                for k, v in browser_config.items()
                if k in BrowserSettings.__annotations__ and v is not None
            }

            # if there is proxy settings, add it to the parameters.
            if proxy_settings:
                valid_browser_params["proxy"] = proxy_settings

            # only create BrowserSettings when there are valid parameters.
            if valid_browser_params:
                browser_settings = BrowserSettings(**valid_browser_params)

        search_config = raw_config.get("search", {})
        search_settings = None
        if search_config:
            search_settings = SearchSettings(**search_config)
        sandbox_config = raw_config.get("sandbox", {})
        if sandbox_config:
            sandbox_settings = SandboxSettings(**sandbox_config)
        else:
            sandbox_settings = SandboxSettings()

        mcp_config = raw_config.get("mcp", {})
        mcp_settings = None
        if mcp_config:
            # Load server configurations from JSON
            mcp_config["servers"] = MCPSettings.load_server_config()
            mcp_settings = MCPSettings(**mcp_config)
        else:
            mcp_settings = MCPSettings(servers=MCPSettings.load_server_config())

        run_flow_config = raw_config.get("runflow")
        if run_flow_config:
            run_flow_settings = RunflowSettings(**run_flow_config)
        else:
            run_flow_settings = RunflowSettings()
        config_dict = {
            "llm": {
                "default": default_settings,
                **{
                    name: {**default_settings, **override_config}
                    for name, override_config in llm_overrides.items()
                },
            },
            "sandbox": sandbox_settings,
            "browser_config": browser_settings,
            "search_config": search_settings,
            "mcp_config": mcp_settings,
            "run_flow_config": run_flow_settings,
        }

        self._config = AppConfig(**config_dict)

        # Store version information
        self._store_config_version()

    def _setup_hot_reload(self):
        """Set up file system monitoring for configuration hot reloading"""
        try:
            config_dir = PROJECT_ROOT / "config"
            if config_dir.exists():
                self._file_observer = Observer()
                event_handler = ConfigurationChangeHandler(self)
                self._file_observer.schedule(
                    event_handler, str(config_dir), recursive=False
                )
                self._file_observer.start()
                self._hot_reload_enabled = True
                print(f"Configuration hot reloading enabled for {config_dir}")
        except Exception as e:
            print(f"Failed to setup configuration hot reloading: {e}")
            self._hot_reload_enabled = False

    def _handle_config_change(self, file_path: Path):
        """Handle configuration file changes"""
        if not self._hot_reload_enabled:
            return

        try:
            print(f"Configuration file changed: {file_path}")

            # Create backup of current configuration
            current_config_backup = self._create_config_backup()

            # Attempt to reload configuration
            if self._reload_configuration():
                print("Configuration reloaded successfully")
                self._notify_change_callbacks("config_reloaded", file_path)
            else:
                print("Configuration reload failed, keeping current configuration")

        except Exception as e:
            print(f"Error handling configuration change: {e}")
            if self._rollback_enabled and current_config_backup:
                self._rollback_configuration(current_config_backup)

    def _reload_configuration(self) -> bool:
        """Reload configuration from files with validation"""
        try:
            # Load new configuration
            raw_config = self._load_config()

            # Validate new configuration before applying
            if self._validation_enabled:
                if not self._validate_configuration(raw_config):
                    return False

            # Parse and apply new configuration
            new_config = self._parse_config(raw_config)

            # Store old config for potential rollback
            old_config = self._config

            # Apply new configuration
            self._config = new_config
            self._store_config_version()

            return True

        except Exception as e:
            print(f"Configuration reload failed: {e}")
            return False

    def _parse_config(self, raw_config: dict) -> AppConfig:
        """Parse raw configuration into AppConfig object"""
        base_llm = raw_config.get("llm", {})
        llm_overrides = {
            k: v for k, v in raw_config.get("llm", {}).items() if isinstance(v, dict)
        }

        default_settings = {
            "model": base_llm.get("model"),
            "base_url": base_llm.get("base_url"),
            "api_key": base_llm.get("api_key"),
            "max_tokens": base_llm.get("max_tokens", 4096),
            "max_input_tokens": base_llm.get("max_input_tokens"),
            "temperature": base_llm.get("temperature", 1.0),
            "api_type": base_llm.get("api_type", ""),
            "api_version": base_llm.get("api_version", ""),
        }

        # Handle browser config
        browser_config = raw_config.get("browser", {})
        browser_settings = None

        if browser_config:
            proxy_config = browser_config.get("proxy", {})
            proxy_settings = None

            if proxy_config and proxy_config.get("server"):
                proxy_settings = ProxySettings(
                    **{
                        k: v
                        for k, v in proxy_config.items()
                        if k in ["server", "username", "password"] and v
                    }
                )

            valid_browser_params = {
                k: v
                for k, v in browser_config.items()
                if k in BrowserSettings.__annotations__ and v is not None
            }

            if proxy_settings:
                valid_browser_params["proxy"] = proxy_settings

            if valid_browser_params:
                browser_settings = BrowserSettings(**valid_browser_params)

        # Handle search config
        search_config = raw_config.get("search", {})
        search_settings = None
        if search_config:
            search_settings = SearchSettings(**search_config)

        # Handle sandbox config
        sandbox_config = raw_config.get("sandbox", {})
        if sandbox_config:
            sandbox_settings = SandboxSettings(**sandbox_config)
        else:
            sandbox_settings = SandboxSettings()

        # Handle MCP config
        mcp_config = raw_config.get("mcp", {})
        mcp_settings = None
        if mcp_config:
            mcp_config["servers"] = MCPSettings.load_server_config()
            mcp_settings = MCPSettings(**mcp_config)
        else:
            mcp_settings = MCPSettings(servers=MCPSettings.load_server_config())

        # Handle run flow config
        run_flow_config = raw_config.get("runflow")
        if run_flow_config:
            run_flow_settings = RunflowSettings(**run_flow_config)
        else:
            run_flow_settings = RunflowSettings()

        # Handle production config
        production_config = raw_config.get("production", {})
        production_settings = None
        if production_config:
            production_settings = ProductionSettings(**production_config)

        config_dict = {
            "llm": {
                "default": default_settings,
                **{
                    name: {**default_settings, **override_config}
                    for name, override_config in llm_overrides.items()
                },
            },
            "sandbox": sandbox_settings,
            "browser_config": browser_settings,
            "search_config": search_settings,
            "mcp_config": mcp_settings,
            "run_flow_config": run_flow_settings,
            "production": production_settings,
            "environment": Environment(raw_config.get("environment", "development")),
            "version": raw_config.get("version"),
            "last_updated": datetime.now(),
        }

        return AppConfig(**config_dict)

    def _validate_configuration(self, raw_config: dict) -> bool:
        """Validate configuration before applying changes"""
        try:
            # Try to parse the configuration
            test_config = self._parse_config(raw_config)

            # Additional validation checks
            if not test_config.llm or "default" not in test_config.llm:
                print("Validation failed: No default LLM configuration found")
                return False

            # Validate LLM settings
            default_llm = test_config.llm["default"]
            if not default_llm.model or not default_llm.api_key:
                print("Validation failed: Default LLM missing required fields")
                return False

            # Validate environment-specific settings
            if test_config.environment == Environment.PRODUCTION:
                if not test_config.production:
                    print(
                        "Validation failed: Production environment requires production settings"
                    )
                    return False

            print("Configuration validation passed")
            return True

        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

    def _create_config_backup(self) -> Optional[AppConfig]:
        """Create a backup of the current configuration"""
        try:
            return self._config.model_copy(deep=True) if self._config else None
        except Exception as e:
            print(f"Failed to create configuration backup: {e}")
            return None

    def _rollback_configuration(self, backup_config: AppConfig):
        """Rollback to a previous configuration"""
        try:
            self._config = backup_config
            print("Configuration rolled back to previous version")
            self._notify_change_callbacks("config_rollback", None)
        except Exception as e:
            print(f"Configuration rollback failed: {e}")

    def _store_config_version(self):
        """Store current configuration version for tracking"""
        try:
            config_data = self._config.model_dump() if self._config else {}
            config_json = json.dumps(config_data, sort_keys=True, default=str)
            checksum = hashlib.sha256(config_json.encode()).hexdigest()

            version = ConfigVersion(
                version=f"v{len(self._config_versions) + 1}",
                timestamp=datetime.now(),
                checksum=checksum,
                environment=(
                    self._config.environment
                    if self._config
                    else Environment.DEVELOPMENT
                ),
                description=f"Configuration version {len(self._config_versions) + 1}",
            )

            self._config_versions.append(version)

            # Keep only last 10 versions
            if len(self._config_versions) > 10:
                self._config_versions = self._config_versions[-10:]

        except Exception as e:
            print(f"Failed to store configuration version: {e}")

    def register_change_callback(self, callback):
        """Register a callback to be called when configuration changes"""
        if callable(callback):
            self._change_callbacks.append(callback)

    def unregister_change_callback(self, callback):
        """Unregister a configuration change callback"""
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)

    def _notify_change_callbacks(self, event_type: str, file_path: Optional[Path]):
        """Notify all registered callbacks about configuration changes"""
        for callback in self._change_callbacks:
            try:
                callback(event_type, file_path, self._config)
            except Exception as e:
                print(f"Error in configuration change callback: {e}")

    def get_config_versions(self) -> List[ConfigVersion]:
        """Get list of configuration versions"""
        return self._config_versions.copy()

    def get_current_version(self) -> Optional[ConfigVersion]:
        """Get current configuration version"""
        return self._config_versions[-1] if self._config_versions else None

    def enable_hot_reload(self):
        """Enable configuration hot reloading"""
        if not self._hot_reload_enabled:
            self._setup_hot_reload()

    def disable_hot_reload(self):
        """Disable configuration hot reloading"""
        if self._file_observer:
            self._file_observer.stop()
            self._file_observer.join()
            self._file_observer = None
        self._hot_reload_enabled = False

    def enable_validation(self):
        """Enable configuration validation"""
        self._validation_enabled = True

    def disable_validation(self):
        """Disable configuration validation"""
        self._validation_enabled = False

    def enable_rollback(self):
        """Enable automatic rollback on configuration errors"""
        self._rollback_enabled = True

    def disable_rollback(self):
        """Disable automatic rollback on configuration errors"""
        self._rollback_enabled = False

    def force_reload(self) -> bool:
        """Force reload configuration from files"""
        return self._reload_configuration()

    def get_config_status(self) -> Dict[str, Any]:
        """Get current configuration status and settings"""
        return {
            "hot_reload_enabled": self._hot_reload_enabled,
            "validation_enabled": self._validation_enabled,
            "rollback_enabled": self._rollback_enabled,
            "current_version": (
                self.get_current_version().model_dump()
                if self.get_current_version()
                else None
            ),
            "total_versions": len(self._config_versions),
            "environment": (
                self._config.environment.value if self._config else "unknown"
            ),
            "last_updated": (
                self._config.last_updated.isoformat()
                if self._config and self._config.last_updated
                else None
            ),
        }

    def __del__(self):
        """Cleanup when Config instance is destroyed"""
        if hasattr(self, "_file_observer") and self._file_observer:
            try:
                self._file_observer.stop()
                self._file_observer.join()
            except Exception:
                pass

    @property
    def llm(self) -> Dict[str, LLMSettings]:
        return self._config.llm

    @property
    def sandbox(self) -> SandboxSettings:
        return self._config.sandbox

    @property
    def browser_config(self) -> Optional[BrowserSettings]:
        return self._config.browser_config

    @property
    def search_config(self) -> Optional[SearchSettings]:
        return self._config.search_config

    @property
    def mcp_config(self) -> MCPSettings:
        """Get the MCP configuration"""
        return self._config.mcp_config

    @property
    def run_flow_config(self) -> RunflowSettings:
        """Get the Run Flow configuration"""
        return self._config.run_flow_config

    @property
    def production(self) -> Optional[ProductionSettings]:
        """Get the production configuration"""
        return self._config.production

    @property
    def environment(self) -> Environment:
        """Get the current environment"""
        return self._config.environment

    @property
    def workspace_root(self) -> Path:
        """Get the workspace root directory"""
        return WORKSPACE_ROOT

    @property
    def root_path(self) -> Path:
        """Get the root path of the application"""
        return PROJECT_ROOT


config = Config()
