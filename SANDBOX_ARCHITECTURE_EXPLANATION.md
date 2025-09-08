# OpenManus Sandbox Architecture - Docker-Based Execution Environment

## Overview

Yes, OpenManus uses **Docker** for its sandbox implementation! The sandbox provides a secure, isolated execution environment for running code and commands safely. Here's how it works:

## Architecture Components

### 1. Core Sandbox (`DockerSandbox`)
**Location**: `app/sandbox/core/sandbox.py`

The main sandbox class that manages Docker containers:

```python
class DockerSandbox:
    """Docker sandbox environment with resource limits and file operations."""

    def __init__(self, config: SandboxSettings, volume_bindings: Dict[str, str]):
        self.config = config or SandboxSettings()
        self.volume_bindings = volume_bindings or {}
        self.client = docker.from_env()  # Docker client
        self.container = None
        self.terminal = None
```

**Key Features**:
- **Resource Limits**: Memory, CPU, and network restrictions
- **File Operations**: Read, write, copy files between host and container
- **Command Execution**: Run shell commands with timeout control
- **Volume Mounting**: Bind host directories to container paths
- **Security**: Path traversal protection and command sanitization

### 2. Terminal Interface (`AsyncDockerizedTerminal`)
**Location**: `app/sandbox/core/terminal.py`

Provides interactive command execution within containers:

```python
class AsyncDockerizedTerminal:
    """Asynchronous terminal for Docker containers."""

    async def run_command(self, cmd: str, timeout: int = None) -> str:
        """Execute command with timeout and return output."""
```

**Features**:
- **Interactive Sessions**: Persistent bash sessions in containers
- **Async Execution**: Non-blocking command execution
- **Timeout Control**: Prevents runaway processes
- **Command Sanitization**: Blocks dangerous operations
- **Environment Management**: Custom environment variables

### 3. Client Interface (`SandboxClient`)
**Location**: `app/sandbox/client.py`

Provides a high-level API for sandbox operations:

```python
class LocalSandboxClient(BaseSandboxClient):
    """Local sandbox client implementation."""

    async def create(self, config: SandboxSettings) -> None:
        """Create and start sandbox container."""

    async def run_command(self, command: str) -> str:
        """Execute command in sandbox."""

    async def read_file(self, path: str) -> str:
        """Read file from container."""
```

## Configuration

### Sandbox Settings
**Location**: `app/config.py`

```python
class SandboxSettings(BaseModel):
    use_sandbox: bool = False              # Enable/disable sandbox
    image: str = "python:3.12-slim"       # Base Docker image
    work_dir: str = "/workspace"           # Container working directory
    memory_limit: str = "512m"             # Memory limit
    cpu_limit: float = 1.0                 # CPU limit (cores)
    timeout: int = 300                     # Command timeout (seconds)
    network_enabled: bool = False          # Network access
```

### Example Configuration
```toml
[sandbox]
use_sandbox = true
image = "python:3.12-slim"
work_dir = "/workspace"
memory_limit = "1g"
cpu_limit = 0.5
timeout = 300
network_enabled = false
```

## How It Works

### 1. Container Creation
```python
# Create container with resource limits
host_config = self.client.api.create_host_config(
    mem_limit=self.config.memory_limit,      # "512m"
    cpu_period=100000,
    cpu_quota=int(100000 * self.config.cpu_limit),  # 50% CPU
    network_mode="none" if not self.config.network_enabled else "bridge",
    binds=self._prepare_volume_bindings(),
)

container = await asyncio.to_thread(
    self.client.api.create_container,
    image=self.config.image,                 # "python:3.12-slim"
    command="tail -f /dev/null",             # Keep container running
    working_dir=self.config.work_dir,        # "/workspace"
    host_config=host_config,
    tty=True,
    detach=True,
)
```

### 2. Interactive Terminal Setup
```python
# Create interactive bash session
startup_command = [
    "bash", "-c",
    f"cd {working_dir} && "
    "PROMPT_COMMAND='' "
    "PS1='$ ' "
    "exec bash --norc --noprofile"
]

exec_data = self.api.exec_create(
    self.container_id,
    startup_command,
    stdin=True, tty=True, stdout=True, stderr=True,
    environment={"TERM": "dumb", "PS1": "$ "}
)
```

### 3. Command Execution Flow
```python
async def run_command(self, command: str, timeout: int = None) -> str:
    # 1. Sanitize command for security
    sanitized_command = self._sanitize_command(command)

    # 2. Send command to container
    full_command = f"{sanitized_command}\necho $?\n"
    self.socket.sendall(full_command.encode())

    # 3. Read output with timeout
    if timeout:
        result = await asyncio.wait_for(read_output(), timeout)
    else:
        result = await read_output()

    return result.strip()
```

### 4. File Operations
```python
# Read file from container
async def read_file(self, path: str) -> str:
    tar_stream, _ = await asyncio.to_thread(
        self.container.get_archive, resolved_path
    )
    content = await self._read_from_tar(tar_stream)
    return content.decode("utf-8")

# Write file to container
async def write_file(self, path: str, content: str) -> None:
    tar_stream = await self._create_tar_stream(
        os.path.basename(path), content.encode("utf-8")
    )
    await asyncio.to_thread(
        self.container.put_archive, parent_dir, tar_stream
    )
```

## Security Features

### 1. Resource Isolation
- **Memory Limits**: Prevent memory exhaustion
- **CPU Limits**: Prevent CPU hogging
- **Network Isolation**: Optional network access control
- **Filesystem Isolation**: Container filesystem separate from host

### 2. Command Sanitization
```python
def _sanitize_command(self, command: str) -> str:
    """Prevent dangerous operations."""
    risky_commands = [
        "rm -rf /", "rm -rf /*", "mkfs", "dd if=/dev/zero",
        ":(){:|:&};:", "chmod -R 777 /", "chown -R"
    ]

    for risky in risky_commands:
        if risky in command.lower():
            raise ValueError(f"Dangerous operation: {risky}")

    return command
```

### 3. Path Traversal Protection
```python
def _safe_resolve_path(self, path: str) -> str:
    """Prevent path traversal attacks."""
    if ".." in path.split("/"):
        raise ValueError("Path contains unsafe patterns")

    return os.path.join(self.config.work_dir, path)
```

## Integration with Tools

### File Operations Tool
**Location**: `app/tool/str_replace_editor.py`

```python
class StrReplaceEditor(BaseTool):
    """File editor with sandbox support."""

    def _get_operator(self) -> FileOperator:
        """Choose between local and sandbox file operations."""
        return (
            self._sandbox_operator
            if config.sandbox.use_sandbox
            else self._local_operator
        )
```

### Python Execution Tool
**Location**: `app/tool/python_execute.py`

```python
async def execute_python_code(code: str) -> str:
    """Execute Python code in sandbox."""
    if config.sandbox.use_sandbox:
        # Write code to sandbox file
        await sandbox_client.write_file("/workspace/temp_script.py", code)

        # Execute in sandbox
        result = await sandbox_client.run_command(
            "python3 /workspace/temp_script.py"
        )
        return result
    else:
        # Execute locally (less secure)
        return exec(code)
```

## Usage Examples

### Basic Sandbox Usage
```python
from app.sandbox.core.sandbox import DockerSandbox
from app.config import SandboxSettings

# Create sandbox configuration
config = SandboxSettings(
    image="python:3.12-slim",
    work_dir="/workspace",
    memory_limit="1g",
    cpu_limit=0.5,
    timeout=300,
    network_enabled=False
)

# Create and use sandbox
async with DockerSandbox(config) as sandbox:
    # Write Python code
    await sandbox.write_file("/workspace/hello.py", """
print("Hello from sandbox!")
import sys
print(f"Python version: {sys.version}")
""")

    # Execute code
    result = await sandbox.run_command("python3 /workspace/hello.py")
    print(result)

    # Read file
    content = await sandbox.read_file("/workspace/hello.py")
    print(f"File content: {content}")
```

### Volume Mounting
```python
# Mount host directory in container
volume_bindings = {
    "/host/data": "/workspace/data"  # host_path: container_path
}

sandbox = DockerSandbox(config, volume_bindings)
await sandbox.create()

# Files in /host/data are accessible at /workspace/data in container
await sandbox.run_command("ls -la /workspace/data")
```

### Client Interface Usage
```python
from app.sandbox.client import SANDBOX_CLIENT

# Initialize sandbox
await SANDBOX_CLIENT.create(config)

# Execute commands
result = await SANDBOX_CLIENT.run_command("pip install requests")
output = await SANDBOX_CLIENT.run_command("python -c 'import requests; print(requests.__version__)'")

# File operations
await SANDBOX_CLIENT.write_file("/workspace/test.txt", "Hello World!")
content = await SANDBOX_CLIENT.read_file("/workspace/test.txt")

# Cleanup
await SANDBOX_CLIENT.cleanup()
```

## Benefits of Docker Sandbox

### 1. **Security**
- **Process Isolation**: Malicious code can't affect host system
- **Resource Limits**: Prevents resource exhaustion attacks
- **Network Isolation**: Optional internet access control
- **Filesystem Protection**: Container filesystem is isolated

### 2. **Consistency**
- **Reproducible Environment**: Same Python version and packages
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Clean State**: Fresh environment for each execution

### 3. **Flexibility**
- **Custom Images**: Use different base images (Python, Node.js, etc.)
- **Package Installation**: Install packages without affecting host
- **Multiple Languages**: Support for any language available in Docker

### 4. **Resource Management**
- **Memory Control**: Prevent out-of-memory conditions
- **CPU Throttling**: Prevent CPU-intensive operations from blocking
- **Timeout Control**: Automatic termination of long-running processes

## Limitations and Considerations

### 1. **Performance Overhead**
- Container startup time (~1-2 seconds)
- Docker API communication overhead
- File I/O through Docker volumes

### 2. **Dependencies**
- Requires Docker to be installed and running
- Docker daemon must be accessible
- Base images need to be pulled initially

### 3. **Platform Differences**
- Windows containers vs Linux containers
- Volume mounting differences between platforms
- Network configuration variations

## Testing

The sandbox includes comprehensive tests:

```python
# Test basic functionality
@pytest.mark.asyncio
async def test_sandbox_python_execution(sandbox):
    """Test Python code execution in sandbox."""
    python_code = """
print("Hello from Python!")
with open('/workspace/test.txt') as f:
    print(f.read())
"""
    await sandbox.write_file("/workspace/test.py", python_code)
    result = await sandbox.run_command("python3 /workspace/test.py")
    assert "Hello from Python!" in result
```

## Summary

OpenManus uses a sophisticated Docker-based sandbox system that provides:

- ✅ **Secure Execution**: Isolated container environment
- ✅ **Resource Control**: Memory, CPU, and network limits
- ✅ **File Operations**: Safe file read/write/copy operations
- ✅ **Command Execution**: Interactive terminal with timeout control
- ✅ **Integration**: Seamless integration with OpenManus tools
- ✅ **Flexibility**: Configurable images, volumes, and settings

The sandbox ensures that code execution is safe, predictable, and isolated from the host system while providing the flexibility needed for various development and analysis tasks.
