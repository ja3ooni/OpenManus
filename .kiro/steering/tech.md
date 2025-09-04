# Technology Stack & Build System

## Core Technologies

- **Python 3.12**: Primary language (supports 3.11-3.13)
- **Pydantic**: Data validation and settings management
- **FastAPI**: Web framework for API endpoints
- **AsyncIO**: Asynchronous programming for agent operations
- **TOML**: Configuration file format
- **Docker**: Containerization and sandboxing

## Key Dependencies

- **LLM Integration**: `openai`, `boto3` (AWS Bedrock)
- **Browser Automation**: `playwright`, `browser-use`, `browsergym`
- **Web Tools**: `crawl4ai`, `beautifulsoup4`, `requests`
- **Search**: `googlesearch-python`, `duckduckgo_search`, `baidusearch`
- **Data Processing**: `numpy`, `datasets`, `pillow`
- **Testing**: `pytest`, `pytest-asyncio`
- **MCP**: `mcp` (Model Context Protocol)

## Package Management

**Recommended**: Use `uv` for faster dependency management
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv venv --python 3.12
source .venv/bin/activate  # Unix/macOS
uv pip install -r requirements.txt
```

**Alternative**: Use conda/pip
```bash
conda create -n open_manus python=3.12
conda activate open_manus
pip install -r requirements.txt
```

## Common Commands

### Development
```bash
# Run main agent
python main.py

# Run with prompt argument
python main.py --prompt "your task here"

# Run multi-agent workflow
python run_flow.py

# Run MCP mode
python run_mcp.py

# Run MCP server
python run_mcp_server.py
```

### Testing
```bash
# Run tests
pytest

# Run async tests
pytest -v tests/

# Pre-commit checks
pre-commit run --all-files
```

### Docker
```bash
# Build image
docker build -t openmanus .

# Run container
docker run -it openmanus
```

### Browser Setup (Optional)
```bash
# Install Playwright browsers
playwright install
```

## Configuration

- Main config: `config/config.toml` (copy from `config.example.toml`)
- MCP config: `config/mcp.json` (copy from `mcp.example.json`)
- Environment: Supports multiple LLM providers via configuration
