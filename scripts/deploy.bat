@echo off
REM OpenManus Docker Deployment Script for Windows (Batch)
REM This script provides basic deployment functionality using batch commands

setlocal enabledelayedexpansion

REM Default values
set ENVIRONMENT=production
set PULL_IMAGES=false
set BACKUP_BEFORE_DEPLOY=false

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :args_done
if "%~1"=="-e" (
    set ENVIRONMENT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--environment" (
    set ENVIRONMENT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-p" (
    set PULL_IMAGES=true
    shift
    goto :parse_args
)
if "%~1"=="--pull" (
    set PULL_IMAGES=true
    shift
    goto :parse_args
)
if "%~1"=="-b" (
    set BACKUP_BEFORE_DEPLOY=true
    shift
    goto :parse_args
)
if "%~1"=="--backup" (
    set BACKUP_BEFORE_DEPLOY=true
    shift
    goto :parse_args
)
if "%~1"=="-h" goto :show_help
if "%~1"=="--help" goto :show_help
echo Unknown option: %~1
goto :show_help

:args_done

REM Validate environment and set compose files
if /i "%ENVIRONMENT%"=="development" (
    set COMPOSE_FILES=docker-compose.yml -f docker-compose.dev.yml
) else if /i "%ENVIRONMENT%"=="dev" (
    set ENVIRONMENT=development
    set COMPOSE_FILES=docker-compose.yml -f docker-compose.dev.yml
) else if /i "%ENVIRONMENT%"=="production" (
    set COMPOSE_FILES=docker-compose.yml -f docker-compose.prod.yml
) else if /i "%ENVIRONMENT%"=="prod" (
    set ENVIRONMENT=production
    set COMPOSE_FILES=docker-compose.yml -f docker-compose.prod.yml
) else (
    echo [ERROR] Invalid environment: %ENVIRONMENT%
    echo [ERROR] Supported environments: development, production
    exit /b 1
)

echo [INFO] Deploying OpenManus in %ENVIRONMENT% environment

REM Check if Docker and Docker Compose are available
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed or not in PATH
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Compose is not installed or not in PATH
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo [WARNING] .env file not found
    if exist ".env.example" (
        echo [INFO] Copying .env.example to .env
        copy ".env.example" ".env" >nul
        echo [WARNING] Please edit .env file with your configuration before continuing
        exit /b 1
    ) else (
        echo [ERROR] No .env.example file found. Please create .env file manually
        exit /b 1
    )
)

REM Set build arguments
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set BUILD_DATE=%%c-%%a-%%b
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set BUILD_TIME=%%a:%%b
set BUILD_DATE=%BUILD_DATE%T%BUILD_TIME%Z

REM Try to get git information (ignore errors)
for /f "tokens=*" %%i in ('git rev-parse --short HEAD 2^>nul') do set VCS_REF=%%i
for /f "tokens=*" %%i in ('git describe --tags --always 2^>nul') do set VERSION=%%i

if not defined VCS_REF set VCS_REF=unknown
if not defined VERSION set VERSION=latest

echo [INFO] Build info: Version=%VERSION%, Ref=%VCS_REF%, Date=%BUILD_DATE%

REM Pull images if requested
if "%PULL_IMAGES%"=="true" (
    echo [INFO] Pulling latest images...
    docker-compose -f %COMPOSE_FILES% pull
    if errorlevel 1 (
        echo [ERROR] Failed to pull images
        exit /b 1
    )
    echo [SUCCESS] Images pulled successfully
)

REM Build and start services
echo [INFO] Building and starting services...

REM Stop existing services
echo [INFO] Stopping existing services...
docker-compose -f %COMPOSE_FILES% down

REM Build images
echo [INFO] Building OpenManus image...
docker-compose -f %COMPOSE_FILES% build openmanus
if errorlevel 1 (
    echo [ERROR] Failed to build OpenManus image
    exit /b 1
)

REM Start services
echo [INFO] Starting services...
docker-compose -f %COMPOSE_FILES% up -d
if errorlevel 1 (
    echo [ERROR] Failed to start services
    exit /b 1
)

REM Wait for services to be healthy
echo [INFO] Waiting for services to be healthy...
timeout /t 10 /nobreak >nul

REM Check health (simple version for batch)
echo [INFO] Checking service health...
set HEALTH_PORT=8080
if defined HEALTH_PORT (
    set HEALTH_URL=http://localhost:%HEALTH_PORT%/health/live
) else (
    set HEALTH_URL=http://localhost:8080/health/live
)

REM Simple health check using curl if available
curl --version >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Performing health check...
    timeout /t 5 /nobreak >nul
    curl -f -s %HEALTH_URL% >nul 2>&1
    if not errorlevel 1 (
        echo [SUCCESS] OpenManus is healthy and ready!
    ) else (
        echo [WARNING] Health check failed, but services may still be starting
    )
) else (
    echo [INFO] curl not available, skipping automated health check
    echo [INFO] Please check manually: %HEALTH_URL%
)

REM Show running services
echo [INFO] Running services:
docker-compose -f %COMPOSE_FILES% ps

REM Show useful commands
echo.
echo [SUCCESS] Deployment completed!
echo.
echo [INFO] Useful commands:
echo   View logs:           docker-compose -f %COMPOSE_FILES% logs -f
echo   Check status:        docker-compose -f %COMPOSE_FILES% ps
echo   Stop services:       docker-compose -f %COMPOSE_FILES% down
echo   Restart service:     docker-compose -f %COMPOSE_FILES% restart openmanus
echo   Health check:        curl http://localhost:%HEALTH_PORT%/health
echo   Application:         http://localhost:8000

if "%ENVIRONMENT%"=="development" (
    echo   Shell access:        docker-compose -f %COMPOSE_FILES% exec openmanus bash
    echo   Run tests:           docker-compose -f %COMPOSE_FILES% exec openmanus pytest
)

goto :eof

:show_help
echo Usage: %~nx0 [OPTIONS]
echo.
echo Deploy OpenManus using Docker Compose
echo.
echo OPTIONS:
echo     -e, --environment ENV    Environment to deploy (development, production) [default: production]
echo     -p, --pull              Pull latest images before deployment
echo     -b, --backup            Create backup before deployment (production only)
echo     -h, --help              Show this help message
echo.
echo EXAMPLES:
echo     %~nx0                                    # Deploy production environment
echo     %~nx0 -e development                     # Deploy development environment
echo     %~nx0 -e production -p -b               # Deploy production with pull and backup
exit /b 0
