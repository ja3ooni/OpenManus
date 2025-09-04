# Project Structure & Organization

## Root Directory Layout

```
OpenManus/
├── app/                    # Core application code
├── config/                 # Configuration files and examples
├── examples/               # Usage examples and benchmarks
├── protocol/               # Protocol definitions (A2A)
├── tests/                  # Test suite
├── workspace/              # Default workspace for agent operations
├── assets/                 # Static assets (logos, images)
├── main.py                 # Single agent entry point
├── run_flow.py            # Multi-agent workflow entry point
├── run_mcp.py             # MCP mode entry point
├── run_mcp_server.py      # MCP server entry point
└── requirements.txt       # Python dependencies
```

## Core Application Structure (`app/`)

### Agent Modules (`app/agent/`)
- `base.py` - Base agent interface and common functionality
- `manus.py` - Main OpenManus agent implementation
- `mcp.py` - MCP (Model Context Protocol) agent
- `browser.py` - Browser automation agent
- `data_analysis.py` - Data analysis specialized agent
- `react.py` - ReAct pattern implementation
- `swe.py` - Software engineering agent
- `toolcall.py` - Tool calling agent

### Tool System (`app/tool/`)
- `base.py` - Base tool interface
- `file_operators.py` - File system operations
- `bash.py` - Shell command execution
- `python_execute.py` - Python code execution
- `web_search.py` - Web search capabilities
- `browser_use_tool.py` - Browser automation tools
- `str_replace_editor.py` - Text editing operations
- `chart_visualization/` - Data visualization tools
- `search/` - Search engine integrations

### Core Services
- `config.py` - Configuration management (singleton pattern)
- `llm.py` - LLM client abstractions
- `bedrock.py` - AWS Bedrock integration
- `logger.py` - Logging configuration
- `schema.py` - Pydantic data models
- `exceptions.py` - Custom exception classes

### Specialized Modules
- `flow/` - Multi-agent workflow orchestration
- `mcp/` - Model Context Protocol implementation
- `prompt/` - Prompt templates and management
- `sandbox/` - Docker-based execution sandbox

## Configuration Structure (`config/`)

- `config.example.toml` - Main configuration template
- `config.example-model-*.toml` - LLM provider specific examples
- `mcp.example.json` - MCP server configuration template

## Architectural Patterns

### Agent Pattern
- All agents inherit from base agent class
- Async/await pattern for all agent operations
- Tool-based architecture with pluggable tools
- Configuration-driven LLM selection

### Tool System
- Tools implement common interface (`base.py`)
- Tools are composable and can be chained
- Each tool handles its own validation and error handling
- Tools support both sync and async operations

### Configuration Management
- Singleton pattern for global configuration
- TOML for main config, JSON for MCP servers
- Environment-specific overrides supported
- Lazy loading with thread-safe initialization

### Workspace Management
- Default workspace in `workspace/` directory
- Configurable workspace root via `config.workspace_root`
- Sandbox isolation for code execution
- File operations scoped to workspace

## Naming Conventions

- **Files**: Snake case (`file_name.py`)
- **Classes**: Pascal case (`ClassName`)
- **Functions/Variables**: Snake case (`function_name`)
- **Constants**: Upper snake case (`CONSTANT_NAME`)
- **Modules**: Short, descriptive names
- **Async functions**: Prefix with `async def`

## Import Organization

1. Standard library imports
2. Third-party imports
3. Local application imports
4. Relative imports (avoid when possible)

## Error Handling

- Custom exceptions in `app/exceptions.py`
- Graceful degradation for optional features
- Comprehensive logging via `app/logger.py`
- Async context managers for resource cleanup
