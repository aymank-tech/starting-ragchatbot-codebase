#!/bin/bash

# Code quality checks for the project

set -e

echo "Running code quality checks..."

# Format check (or apply formatting with --fix flag)
if [ "$1" = "--fix" ]; then
    echo ""
    echo "=== Applying black formatting ==="
    uv run black backend/
else
    echo ""
    echo "=== Checking black formatting ==="
    uv run black --check backend/
fi

echo ""
echo "All quality checks passed!"
