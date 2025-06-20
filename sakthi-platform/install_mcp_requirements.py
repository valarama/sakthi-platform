#!/usr/bin/env python3
"""
Install additional requirements for Dynamic MCP System
"""

import subprocess
import sys

def install_requirements():
    """Install requirements for dynamic MCP system"""
    
    requirements = [
        "jinja2>=3.0.0",
        "aiofiles",
        "python-multipart"
    ]
    
    print("🔧 Installing Dynamic MCP requirements...")
    
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✅ Installed: {req}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {req}: {e}")
    
    print("✅ Requirements installation completed!")

if __name__ == "__main__":
    install_requirements()
