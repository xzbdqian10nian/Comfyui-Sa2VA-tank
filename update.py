#!/usr/bin/env python3
"""
ComfyUI-Sa2VA Update Script

This script handles updating the node and automatically configures Git
to prevent update conflicts.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True, capture_output=True):
    """Run a shell command with proper error handling."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=capture_output,
            text=True,
            check=check
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        return False, "", str(e)


def configure_git():
    """Configure Git to prevent pull conflicts."""
    print("⚙️  Configuring Git for smooth updates...")
    
    # Set pull.rebase to true for this repository
    success, stdout, stderr = run_command("git config pull.rebase true")
    
    if success:
        print("✅ Git pull strategy configured (rebase)")
        return True
    else:
        # Check if this is a git repository
        is_git_repo, _, _ = run_command("git rev-parse --git-dir", check=False)
        if not is_git_repo:
            print("⚠️  Not a git repository (normal for manual installs)")
            return True
        else:
            print(f"⚠️  Could not configure Git: {stderr}")
            return False


def update_node():
    """Update the node from remote repository."""
    print("\n🔄 Updating ComfyUI-Sa2VA node...")
    
    # First, configure Git
    configure_git()
    
    # Check if we're in a git repository
    is_git_repo, _, _ = run_command("git rev-parse --git-dir", check=False)
    if not is_git_repo:
        print("\n❌ Not a git repository!")
        print("   This script only works with git installations.")
        print("   If you installed manually, please reinstall from:")
        print("   https://github.com/xzbdqian10nian/Comfyui-Sa2VA-tank.git")
        return False
    
    # Fetch latest changes
    print("\n📥 Fetching latest changes...")
    success, stdout, stderr = run_command("git fetch origin")
    if not success:
        print(f"❌ Failed to fetch: {stderr}")
        return False
    
    # Pull with rebase (now configured)
    print("⬇️  Pulling latest changes...")
    success, stdout, stderr = run_command("git pull --rebase")
    
    if success:
        print("\n✅ Node updated successfully!")
        print(stdout)
        return True
    else:
        print(f"\n❌ Update failed: {stderr}")
        print("\n💡 If you have local changes, you may need to:")
        print("   git stash")
        print("   git pull")
        print("   git stash pop")
        return False


def main():
    """Main update function."""
    print("🚀 ComfyUI-Sa2VA Update Script")
    print("=" * 50)
    
    if update_node():
        print("\n" + "=" * 50)
        print("🎉 Update complete!")
        print("=" * 50)
        print("\n📋 Next Steps:")
        print("1. Restart ComfyUI to load the updated nodes")
        print("2. Check the changelog for new features")
        return 0
    else:
        print("\n" + "=" * 50)
        print("❌ Update failed")
        print("=" * 50)
        return 1


if __name__ == "__main__":
    sys.exit(main())
