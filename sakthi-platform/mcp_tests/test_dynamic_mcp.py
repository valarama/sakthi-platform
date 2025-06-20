#!/usr/bin/env python3
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
