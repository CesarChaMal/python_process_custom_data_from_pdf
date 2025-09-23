#!/bin/bash
# Quick Python environment test

echo "🔍 Testing Python installations..."

echo "1. Testing 'python':"
if command -v python &> /dev/null; then
    python --version 2>&1 && echo "✅ python works"
else
    echo "❌ python not found"
fi

echo "2. Testing 'python3':"
if command -v python3 &> /dev/null; then
    python3 --version 2>&1 && echo "✅ python3 works"
else
    echo "❌ python3 not found"
fi

echo "3. Testing 'py':"
if command -v py &> /dev/null; then
    py --version 2>&1 && echo "✅ py works"
else
    echo "❌ py not found"
fi

echo "4. Environment info:"
echo "PATH: $PATH"
echo "OSTYPE: $OSTYPE"
echo "PWD: $PWD"