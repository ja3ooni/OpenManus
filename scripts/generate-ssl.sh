#!/bin/bash
# Generate self-signed SSL certificates for development

set -e

# Create SSL directory
mkdir -p config/ssl

# Generate private key
openssl genrsa -out config/ssl/key.pem 2048

# Generate certificate signing request
openssl req -new -key config/ssl/key.pem -out config/ssl/cert.csr -subj "/C=US/ST=State/L=City/O=OpenManus/OU=Development/CN=localhost"

# Generate self-signed certificate
openssl x509 -req -days 365 -in config/ssl/cert.csr -signkey config/ssl/key.pem -out config/ssl/cert.pem

# Clean up CSR
rm config/ssl/cert.csr

# Set proper permissions
chmod 600 config/ssl/key.pem
chmod 644 config/ssl/cert.pem

echo "SSL certificates generated successfully:"
echo "  Certificate: config/ssl/cert.pem"
echo "  Private key: config/ssl/key.pem"
echo ""
echo "Note: These are self-signed certificates for development only."
echo "For production, use certificates from a trusted CA."
