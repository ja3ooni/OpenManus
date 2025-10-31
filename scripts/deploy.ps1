# OpenManus Docker Deployment Script for Windows
# This script helps deploy OpenManus in different environments using PowerShell

param(
    [string]$Environment = "production",
    [switch]$Pull = $false,
    [switch]$Backup = $false,
    [switch]$Help = $false
)

# Function to show usage
function Show-Usage {
    Write-Host @"
Usage: .\deploy.ps1 [OPTIONS]

Deploy OpenManus using Docker Compose

OPTIONS:
    -Environment ENV    Environment to deploy (development, production) [default: production]
    -Pull              Pull latest images before deployment
    -Backup            Create backup before deployment (production only)
    -Help              Show this help message

EXAMPLES:
    .\deploy.ps1                                    # Deploy production environment
    .\deploy.ps1 -Environment development           # Deploy development environment
    .\deploy.ps1 -Environment production -Pull -Backup  # Deploy production with pull and backup
"@
}

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Show help if requested
if ($Help) {
    Show-Usage
    exit 0
}

# Validate environment
$ComposeFiles = ""
switch ($Environment.ToLower()) {
    { $_ -in @("development", "dev") } {
        $Environment = "development"
        $ComposeFiles = "docker-compose.yml", "docker-compose.dev.yml"
    }
    { $_ -in @("production", "prod") } {
        $Environment = "production"
        $ComposeFiles = "docker-compose.yml", "docker-compose.prod.yml"
    }
    default {
        Write-Error "Invalid environment: $Environment"
        Write-Error "Supported environments: development, production"
        exit 1
    }
}

Write-Status "Deploying OpenManus in $Environment environment"

# Check if Docker and Docker Compose are available
try {
    $null = Get-Command docker -ErrorAction Stop
} catch {
    Write-Error "Docker is not installed or not in PATH"
    exit 1
}

try {
    $null = Get-Command docker-compose -ErrorAction Stop
} catch {
    Write-Error "Docker Compose is not installed or not in PATH"
    exit 1
}

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Warning ".env file not found"
    if (Test-Path ".env.example") {
        Write-Status "Copying .env.example to .env"
        Copy-Item ".env.example" ".env"
        Write-Warning "Please edit .env file with your configuration before continuing"
        exit 1
    } else {
        Write-Error "No .env.example file found. Please create .env file manually"
        exit 1
    }
}

# Set build arguments
$BuildDate = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$VcsRef = "unknown"
$Version = "latest"

# Try to get git information
try {
    $VcsRef = (git rev-parse --short HEAD 2>$null)
    $Version = (git describe --tags --always 2>$null)
} catch {
    # Git not available or not a git repository
}

$env:BUILD_DATE = $BuildDate
$env:VCS_REF = $VcsRef
$env:VERSION = $Version

Write-Status "Build info: Version=$Version, Ref=$VcsRef, Date=$BuildDate"

# Create backup if requested (production only)
if ($Backup -and $Environment -eq "production") {
    Write-Status "Creating backup before deployment..."

    # Check if backup service is available
    try {
        $BackupStatus = docker-compose -f $ComposeFiles ps backup 2>$null
        if ($LASTEXITCODE -eq 0) {
            docker-compose -f $ComposeFiles exec backup /backup.sh
            Write-Success "Backup completed"
        } else {
            Write-Warning "Backup service not available, skipping backup"
        }
    } catch {
        Write-Warning "Backup service not available, skipping backup"
    }
}

# Pull images if requested
if ($Pull) {
    Write-Status "Pulling latest images..."
    $PullArgs = @("-f") + $ComposeFiles + @("pull")
    & docker-compose @PullArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Images pulled successfully"
    } else {
        Write-Error "Failed to pull images"
        exit 1
    }
}

# Build and start services
Write-Status "Building and starting services..."

# Stop existing services
Write-Status "Stopping existing services..."
$DownArgs = @("-f") + $ComposeFiles + @("down")
& docker-compose @DownArgs

# Build images
Write-Status "Building OpenManus image..."
$BuildArgs = @("-f") + $ComposeFiles + @("build", "openmanus")
& docker-compose @BuildArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to build OpenManus image"
    exit 1
}

# Start services
Write-Status "Starting services..."
$UpArgs = @("-f") + $ComposeFiles + @("up", "-d")
& docker-compose @UpArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to start services"
    exit 1
}

# Wait for services to be healthy
Write-Status "Waiting for services to be healthy..."
Start-Sleep -Seconds 10

# Check health
Write-Status "Checking service health..."
$HealthPort = $env:HEALTH_PORT
if (-not $HealthPort) { $HealthPort = "8080" }
$HealthCheckUrl = "http://localhost:$HealthPort/health/live"

$HealthCheckPassed = $false
for ($i = 1; $i -le 30; $i++) {
    try {
        $Response = Invoke-WebRequest -Uri $HealthCheckUrl -TimeoutSec 5 -UseBasicParsing
        if ($Response.StatusCode -eq 200) {
            Write-Success "OpenManus is healthy and ready!"
            $HealthCheckPassed = $true
            break
        }
    } catch {
        # Health check failed, continue trying
    }

    if ($i -eq 30) {
        Write-Error "Health check failed after 30 attempts"
        Write-Error "Check logs with: docker-compose -f $($ComposeFiles -join ' -f ') logs openmanus"
        exit 1
    }

    Write-Status "Waiting for health check... (attempt $i/30)"
    Start-Sleep -Seconds 2
}

# Show running services
Write-Status "Running services:"
$PsArgs = @("-f") + $ComposeFiles + @("ps")
& docker-compose @PsArgs

# Show useful commands
Write-Success "Deployment completed successfully!"
Write-Host ""
Write-Status "Useful commands:"
$ComposeFilesStr = ($ComposeFiles | ForEach-Object { "-f $_" }) -join " "
Write-Host "  View logs:           docker-compose $ComposeFilesStr logs -f"
Write-Host "  Check status:        docker-compose $ComposeFilesStr ps"
Write-Host "  Stop services:       docker-compose $ComposeFilesStr down"
Write-Host "  Restart service:     docker-compose $ComposeFilesStr restart openmanus"
Write-Host "  Health check:        curl http://localhost:$HealthPort/health"

$AppPort = $env:OPENMANUS_PORT
if (-not $AppPort) { $AppPort = "8000" }
Write-Host "  Application:         http://localhost:$AppPort"

if ($Environment -eq "development") {
    Write-Host "  Shell access:        docker-compose $ComposeFilesStr exec openmanus bash"
    Write-Host "  Run tests:           docker-compose $ComposeFilesStr exec openmanus pytest"
}
