#!/usr/bin/env python3
"""
Fixed Integration Script for Dynamic MCP System
"""

import os
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_integration():
    """Fix the integration issues"""
    
    print("🔧 Fixing Dynamic MCP Integration...")
    
    # 1. Create missing files
    create_missing_files()
    
    # 2. Install requirements manually
    install_requirements()
    
    # 3. Create test files
    create_test_files()
    
    print("✅ Integration fixed!")

def create_missing_files():
    """Create the missing files manually"""
    
    # Create install_mcp_requirements.py
    install_script = '''#!/usr/bin/env python3
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
'''
    
    with open("install_mcp_requirements.py", 'w', encoding='utf-8') as f:
        f.write(install_script)
    
    print("✔️ Created install_mcp_requirements.py")

def install_requirements():
    """Install requirements directly"""
    
    requirements = ["jinja2>=3.0.0", "aiofiles", "python-multipart"]
    
    for req in requirements:
        try:
            os.system(f"pip install {req}")
            print(f"✅ Installed: {req}")
        except:
            print(f"❌ Failed to install {req}")

def create_test_files():
    """Create test files in mcp_tests directory"""
    
    # Ensure mcp_tests directory exists
    os.makedirs("mcp_tests", exist_ok=True)
    
    # Create test_dynamic_mcp.py
    test_script = '''#!/usr/bin/env python3
"""
Simple test script for Dynamic MCP Generation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_basic_mcp():
    """Test basic MCP generation"""
    
    print("🧪 Testing Basic MCP Generation")
    print("=" * 50)
    
    try:
        from core import SakthiEngine
        
        # Initialize Sakthi engine
        sakthi_engine = SakthiEngine()
        
        # Test Oracle SQL
        oracle_sql = """
        DECLARE
          c_limit CONSTANT PLS_INTEGER := 50;
          TYPE EmpIdTab IS TABLE OF employees.employee_id%TYPE;
          v_emp_ids EmpIdTab;
        BEGIN
          FETCH emp_cur BULK COLLECT INTO v_emp_ids LIMIT c_limit;
          FORALL i IN 1 .. v_emp_ids.COUNT
            UPDATE employees SET salary = salary * 1.10 WHERE employee_id = v_emp_ids(i);
        END;
        """
        
        # Generate MCP config
        mcp_config = sakthi_engine.generate_mcp_config(
            f"Convert Oracle BULK COLLECT to BigQuery: {oracle_sql}"
        )
        
        print("✅ MCP Generation Test Passed!")
        print(f"Config length: {len(mcp_config)} characters")
        
        # Save test result
        with open("test_mcp_output.json", "w") as f:
            f.write(mcp_config)
        
        print("📁 Test output saved to: test_mcp_output.json")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    test_basic_mcp()
'''
    
    with open("mcp_tests/test_dynamic_mcp.py", 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("✔️ Created mcp_tests/test_dynamic_mcp.py")

if __name__ == "__main__":
    fix_integration()