@echo off
REM Generate self-signed SSL certificates for development on Windows

echo Generating SSL certificates for OpenManus development...

REM Create SSL directory
if not exist "config\ssl" mkdir "config\ssl"

REM Check if OpenSSL is available
openssl version >nul 2>&1
if errorlevel 1 (
    echo ERROR: OpenSSL is not installed or not in PATH
    echo Please install OpenSSL from: https://slproweb.com/products/Win32OpenSSL.html
    echo Or use Git Bash which includes OpenSSL
    pause
    exit /b 1
)

REM Generate private key
echo Generating private key...
openssl genrsa -out config\ssl\key.pem 2048
if errorlevel 1 (
    echo ERROR: Failed to generate private key
    pause
    exit /b 1
)

REM Generate certificate signing request
echo Generating certificate signing request...
openssl req -new -key config\ssl\key.pem -out config\ssl\cert.csr -subj "/C=US/ST=State/L=City/O=OpenManus/OU=Development/CN=localhost"
if errorlevel 1 (
    echo ERROR: Failed to generate certificate signing request
    pause
    exit /b 1
)

REM Generate self-signed certificate
echo Generating self-signed certificate...
openssl x509 -req -days 365 -in config\ssl\cert.csr -signkey config\ssl\key.pem -out config\ssl\cert.pem
if errorlevel 1 (
    echo ERROR: Failed to generate certificate
    pause
    exit /b 1
)

REM Clean up CSR
del config\ssl\cert.csr

echo.
echo SSL certificates generated successfully:
echo   Certificate: config\ssl\cert.pem
echo   Private key: config\ssl\key.pem
echo.
echo Note: These are self-signed certificates for development only.
echo For production, use certificates from a trusted CA.
echo.
pause
