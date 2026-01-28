#!/bin/bash
# ComfyUI-Sa2VA Update Script (Shell version)
# This script handles updating the node and automatically configures Git

set -e

echo "🚀 ComfyUI-Sa2VA Update Script"
echo "=================================================="

# Configure Git to prevent conflicts
echo ""
echo "⚙️  Configuring Git for smooth updates..."
git config pull.rebase true 2>/dev/null || echo "⚠️  Could not configure Git (may not be a git repo)"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo ""
    echo "❌ Not a git repository!"
    echo "   This script only works with git installations."
    echo "   If you installed manually, please reinstall from:"
    echo "   https://github.com/xzbdqian10nian/Comfyui-Sa2VA-tank.git"
    exit 1
fi

# Fetch latest changes
echo ""
echo "📥 Fetching latest changes..."
git fetch origin

# Pull with rebase
echo "⬇️  Pulling latest changes..."
if git pull --rebase; then
    echo ""
    echo "=================================================="
    echo "🎉 Update complete!"
    echo "=================================================="
    echo ""
    echo "📋 Next Steps:"
    echo "1. Restart ComfyUI to load the updated nodes"
    echo "2. Check the changelog for new features"
else
    echo ""
    echo "=================================================="
    echo "❌ Update failed"
    echo "=================================================="
    echo ""
    echo "💡 If you have local changes, you may need to:"
    echo "   git stash"
    echo "   git pull"
    echo "   git stash pop"
    exit 1
fi
