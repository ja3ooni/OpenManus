---
title: UV Package Manager Best Practices
inclusion: always
---

# UV Package Manager Best Practices

UV is a fast Python package installer and resolver, written in Rust. It's designed as a drop-in replacement for pip and pip-tools, with significantly better performance.

## Core Principles
- **ALWAYS use UV instead of pip** - UV is faster, more reliable, and has better dependency resolution
- **NEVER use global Python package installation** - Always work within virtual environments
- **Use UV for all Python package management tasks** - Installation, updates, dependency resolution

## Virtual Environment Management

### Creating Virtual Environments
```bash
# Create virtual environment (preferred method)
uv venv .venv

# Create with specific Python version
uv venv --python 3.12 .venv

# Create in custom location
uv venv /path/to/venv
```

### Activation
```bash
# Windows
.venv\Scripts\activate

# Unix/macOS/Linux
source .venv/bin/activate
```

## Package Installation

### Basic Installation
```bash
# Install single package
uv pip install package_name

# Install with version constraints
uv pip install "package_name>=1.0,<2.0"

# Install from requirements file
uv pip install -r requirements.txt

# Install in editable mode (development)
uv pip install -e .
```

### Advanced Installation
```bash
# Install with extras
uv pip install "package_name[extra1,extra2]"

# Install from git repository
uv pip install git+https://github.com/user/repo.git

# Install from local path
uv pip install ./local/package/path
```

## Dependency Management

### Requirements Files
```bash
# Generate requirements.txt
uv pip freeze > requirements.txt

# Install from requirements with exact versions
uv pip install -r requirements.txt

# Upgrade packages
uv pip install --upgrade package_name
uv pip install --upgrade -r requirements.txt
```

### Project-based Management (pyproject.toml)
```bash
# Sync dependencies from pyproject.toml
uv sync

# Add new dependency
uv add package_name

# Add development dependency
uv add --dev pytest

# Remove dependency
uv remove package_name

# Update dependencies
uv sync --upgrade
```

## Performance Benefits
- **10-100x faster** than pip for most operations
- **Better dependency resolution** - Resolves conflicts more intelligently
- **Parallel downloads** - Downloads packages concurrently
- **Better caching** - Efficient package caching system
- **Cross-platform consistency** - Same behavior across all platforms

## Migration from pip

### Replace Common pip Commands
```bash
# Old pip commands → New UV commands
pip install package          → uv pip install package
pip install -r requirements  → uv pip install -r requirements
pip freeze                   → uv pip freeze
pip list                     → uv pip list
pip uninstall package        → uv pip uninstall package
pip show package             → uv pip show package
```

### Environment Variables
```bash
# UV respects pip environment variables
export UV_INDEX_URL=https://custom.pypi.org/simple/
export UV_EXTRA_INDEX_URL=https://extra.pypi.org/simple/
```

## Best Practices for Development

### Project Setup
1. **Always start with UV virtual environment**:
   ```bash
   uv venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```

2. **Install project dependencies**:
   ```bash
   uv pip install -r requirements.txt
   uv pip install -e .  # If developing the package
   ```

3. **Keep requirements files updated**:
   ```bash
   uv pip freeze > requirements.txt
   ```

### CI/CD Integration
```yaml
# GitHub Actions example
- name: Set up Python with UV
  run: |
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv venv .venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
```

### Docker Integration
```dockerfile
# Install UV in Docker
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Use UV for package installation
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install -r requirements.txt
```

## Troubleshooting

### Common Issues
- **UV not found**: Ensure UV is installed and in PATH
- **Permission errors**: Use virtual environments, avoid global installation
- **Dependency conflicts**: UV provides better error messages than pip
- **Cache issues**: Clear cache with `uv cache clean`

### Installation
```bash
# Install UV (Unix/macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install UV (Windows)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install via pip (if needed)
pip install uv
```

## Integration with IDEs
- **VS Code**: UV virtual environments are automatically detected
- **PyCharm**: Configure interpreter to use UV-created virtual environment
- **Command line**: Always activate virtual environment before running Python

## Security Considerations
- UV uses the same security model as pip
- Verify package sources and checksums
- Use private package indexes when needed
- Keep UV updated for security patches

Remember: UV is designed to be a drop-in replacement for pip with better performance and dependency resolution. Always prefer UV over pip for any Python package management task.
