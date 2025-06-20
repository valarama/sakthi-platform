#!/usr/bin/env python3
"""
Integration Script for Dynamic MCP System
Sets up the complete dynamic MCP generation pipeline
"""

import os
import json
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicMCPSetup:
    """Setup dynamic MCP system integration"""
    
    def __init__(self, platform_root: str = "."):
        self.platform_root = Path(platform_root)
        self.setup_completed = False
    
    def run_complete_setup(self):
        """Run complete setup for dynamic MCP system"""
        
        logger.info("🚀 Setting up Dynamic MCP System...")
        
        try:
            # Step 1: Create directory structure
            self._create_directories()
            
            # Step 2: Setup template system
            self._setup_templates()
            
            # Step 3: Update existing files
            self._update_backend_integration()
            
            # Step 4: Create test files
            self._create_test_files()
            
            # Step 5: Setup requirements
            self._setup_requirements()
            
            # Step 6: Verify installation
            self._verify_installation()
            
            self.setup_completed = True
            logger.info("✅ Dynamic MCP System setup completed successfully!")
            
            # Print usage instructions
            self._print_usage_instructions()
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {str(e)}")
            raise
    
    def _create_directories(self):
        """Create required directory structure"""
        
        directories = [
            "mcp_templates",
            "generated_mcp",
            "mcp_tests",
            "mcp_logs"
        ]
        
        for directory in directories:
            dir_path = self.platform_root / directory
            dir_path.mkdir(exist_ok=True)
            logger.info(f"✔️ Created directory: {dir_path}")
            
            # Create __init__.py for Python packages
            if directory not in ["generated_mcp", "mcp_logs"]:
                (dir_path / "__init__.py").touch()
    
    def _setup_templates(self):
        """Setup template files and configurations"""
        
        # Create the main config file (already done in artifacts)
        config_file = self.platform_root / "mcp_templates" / "oracle_bigquery_config.json"
        if not config_file.exists():
            logger.info("Creating MCP template configuration...")
            # The config is already created in the artifacts above
        
        # Create additional template files
        self._create_prompt_templates()
        self._create_validation_templates()
    
    def _create_prompt_templates(self):
        """Create additional prompt template files"""
        
        prompt_templates = {
            "oracle_analysis_prompts.json": {
                "bulk_collect_analysis": {
                    "template": "Analyze this Oracle BULK COLLECT pattern:\n\n{{ oracle_code }}\n\nProvide:\n1. What data is being processed\n2. Batch size and performance implications\n3. BigQuery equivalent approach\n4. Error handling considerations",
                    "max_tokens": 400,
                    "temperature": 0.1
                },
                "forall_analysis": {
                    "template": "Analyze this Oracle FORALL statement:\n\n{{ oracle_code }}\n\nProvide:\n1. Bulk operation pattern\n2. Exception handling approach\n3. BigQuery DML equivalent\n4. Performance optimization tips",
                    "max_tokens": 300,
                    "temperature": 0.1
                },
                "cursor_analysis": {
                    "template": "Analyze this Oracle cursor usage:\n\n{{ oracle_code }}\n\nProvide:\n1. Cursor pattern and purpose\n2. Data access pattern\n3. BigQuery loop or set-based alternative\n4. Memory and performance considerations",
                    "max_tokens": 350,
                    "temperature": 0.1
                }
            },
            
            "bigquery_generation_prompts.json": {
                "procedure_conversion": {
                    "template": "Convert this Oracle procedure to BigQuery:\n\nORACLE CODE:\n{{ oracle_code }}\n\nGenerate executable BigQuery SQL with:\n1. Proper variable declarations\n2. Loop constructs for batch operations\n3. Error handling\n4. Comments explaining conversion choices\n\nFocus on BigQuery best practices and performance.",
                    "max_tokens": 1000,
                    "temperature": 0.2
                },
                "optimization_suggestions": {
                    "template": "Suggest BigQuery optimizations for this converted code:\n\nCONVERTED CODE:\n{{ bigquery_code }}\n\nProvide:\n1. Partitioning strategies\n2. Clustering recommendations\n3. Query optimization tips\n4. Cost optimization suggestions",
                    "max_tokens": 500,
                    "temperature": 0.3
                }
            }
        }
        
        for filename, content in prompt_templates.items():
            template_file = self.platform_root / "mcp_templates" / filename
            with open(template_file, 'w') as f:
                json.dump(content, f, indent=2)
            logger.info(f"✔️ Created prompt template: {filename}")
    
    def _create_validation_templates(self):
        """Create validation template files"""
        
        validation_config = {
            "mcp_validation_rules": {
                "required_sections": [
                    "mcpVersion",
                    "server",
                    "capabilities", 
                    "resources",
                    "tools",
                    "prompts",
                    "metadata"
                ],
                "server_validation": {
                    "name_pattern": "^sakthi-[a-z0-9\\-]+$",
                    "version_pattern": "^\\d+\\.\\d+\\.\\d+$",
                    "description_min_length": 10
                },
                "resource_validation": {
                    "uri_pattern": "^sakthi://[a-z0-9/\\-_]+$",
                    "required_fields": ["uri", "name", "description", "mimeType"],
                    "mime_types": ["application/sql", "application/json", "text/plain"]
                },
                "tool_validation": {
                    "name_pattern": "^[a-z0-9_]+$",
                    "required_fields": ["name", "description", "inputSchema"],
                    "input_schema_required": ["type", "properties"]
                },
                "prompt_validation": {
                    "name_pattern": "^[a-z0-9_]+$",
                    "required_fields": ["name", "description", "arguments", "template"],
                    "template_min_length": 50
                },
                "quality_thresholds": {
                    "minimum_score": 70,
                    "warning_score": 85,
                    "excellent_score": 95
                }
            }
        }
        
        validation_file = self.platform_root / "mcp_templates" / "validation_config.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_config, f, indent=2)
        logger.info("✔️ Created validation configuration")
    
    def _update_backend_integration(self):
        """Update backend files for integration"""
        
        # Update main.py to use dynamic MCP
        main_py_path = self.platform_root / "backend" / "main.py"
        if main_py_path.exists():
            self._update_main_py(main_py_path)
        
        # Update core.py with new method
        core_py_path = self.platform_root / "core.py"
        if core_py_path.exists():
            self._update_core_py(core_py_path)
    
    def _update_main_py(self, main_py_path: Path):
        """Update main.py with dynamic MCP integration"""
        
        # Read current content
        with open(main_py_path, 'r') as f:
            content = f.read()
        
        # Add import for dynamic MCP
        if "from mcp_engine import" not in content:
            import_line = "\n# Dynamic MCP Integration\nfrom mcp_engine import SakthiMCPIntegration\nimport asyncio\n"
            
            # Find where to insert the import
            if "from core import" in content:
                content = content.replace("from core import", import_line + "from core import")
            else:
                # Add after other imports
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('app = FastAPI'):
                        lines.insert(i, import_line)
                        break
                content = '\n'.join(lines)
        
        # Add MCP integration initialization
        if "mcp_integration = SakthiMCPIntegration" not in content:
            init_code = """
# Initialize MCP Integration
mcp_integration = None

def get_mcp_integration():
    global mcp_integration
    if mcp_integration is None:
        from core import SakthiEngine
        sakthi_engine = SakthiEngine()
        mcp_integration = SakthiMCPIntegration(sakthi_engine)
    return mcp_integration
"""
            # Insert before the main endpoint
            content = content.replace("@app.post(\"/api/v1/convert/oracle-to-bigquery\")", 
                                    init_code + "\n@app.post(\"/api/v1/convert/oracle-to-bigquery\")")
        
        # Update the endpoint to use dynamic MCP
        if "# Generate MCP config using the method we added to SakthiEngine" in content:
            old_mcp_line = "mcp_config = sakthi.generate_mcp_config(f\"{intent}: {oracle_sql}\")"
            new_mcp_code = """
        # Generate dynamic MCP config with LLM analysis
        mcp_integration = get_mcp_integration()
        mcp_result = await mcp_integration.process_oracle_file_to_mcp(
            oracle_sql, file.filename, include_llm_analysis=True
        )
        
        if mcp_result["success"]:
            mcp_config = mcp_result["mcp_config"]
            mcp_analysis = mcp_result["llm_analysis"]
            mcp_validation = mcp_result["validation"]
        else:
            mcp_config = "{\"error\": \"MCP generation failed\"}"
            mcp_analysis = None
            mcp_validation = None"""
            
            content = content.replace(old_mcp_line, new_mcp_code)
        
        # Update return statement to include MCP analysis
        if "\"mcp_config\": mcp_config," in content and "\"mcp_analysis\":" not in content:
            content = content.replace(
                "\"mcp_config\": mcp_config,",
                "\"mcp_config\": mcp_config,\n            \"mcp_analysis\": mcp_analysis,\n            \"mcp_validation\": mcp_validation,"
            )
        
        # Write updated content
        with open(main_py_path, 'w') as f:
            f.write(content)
        
        logger.info("✔️ Updated main.py with dynamic MCP integration")
    
    def _update_core_py(self, core_py_path: Path):
        """Update core.py with enhanced MCP method"""
        
        # Read current content
        with open(core_py_path, 'r') as f:
            content = f.read()
        
        # Add enhanced MCP method if not exists
        if "generate_enhanced_mcp_config" not in content:
            enhanced_method = '''
    def generate_enhanced_mcp_config(self, input_text: str, oracle_sql: str = "", output_format: str = "json") -> str:
        """Generate comprehensive MCP configuration with Oracle context"""
        
        # Parse with existing Sakthi logic
        intent = self.parser.parse(input_text)
        
        # Enhanced MCP config with dynamic analysis
        mcp_config = {
            "mcpVersion": "2024-11-05",
            "server": {
                "name": f"sakthi-oracle-{hash(oracle_sql) % 10000}",
                "version": "1.0.0",
                "description": f"Oracle to BigQuery conversion - {intent.intent_type.value}"
            },
            "capabilities": {
                "resources": {"subscribe": True, "listChanged": True},
                "tools": {"listChanged": True},
                "prompts": {"listChanged": True},
                "logging": {"level": "info"}
            },
            "resources": [
                {
                    "uri": f"sakthi://oracle/dynamic/{hash(oracle_sql) % 10000}",
                    "name": f"oracle_conversion_{intent.intent_type.value}",
                    "description": f"Dynamic Oracle analysis for {intent.intent_type.value}",
                    "mimeType": "application/sql",
                    "annotations": {
                        "source_type": "oracle_plsql",
                        "conversion_target": "bigquery_sql",
                        "intent_confidence": intent.confidence,
                        "detected_entities": intent.entities,
                        "processing_params": intent.parameters
                    }
                }
            ],
            "tools": [
                {
                    "name": "dynamic_oracle_converter",
                    "description": f"Convert Oracle code with {intent.intent_type.value} intent",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "oracle_code": {"type": "string", "description": "Oracle PL/SQL code"},
                            "conversion_intent": {"type": "string", "default": intent.intent_type.value},
                            "quality_level": {"type": "string", "enum": ["basic", "detailed", "comprehensive"], "default": "detailed"}
                        },
                        "required": ["oracle_code"]
                    }
                }
            ],
            "prompts": [
                {
                    "name": "dynamic_oracle_expert",
                    "description": "Expert Oracle to BigQuery conversion with dynamic analysis",
                    "arguments": [
                        {"name": "oracle_code", "description": "Oracle PL/SQL code", "required": True},
                        {"name": "user_intent", "description": "User conversion intent", "required": False}
                    ],
                    "template": f"Convert this Oracle code to BigQuery based on intent: {intent.intent_type.value}\\n\\nOracle Code: {{oracle_code}}\\n\\nUser Intent: {{user_intent}}\\n\\nProvide optimized BigQuery conversion with detailed explanations."
                }
            ],
            "metadata": {
                "generated_by": "sakthi-enhanced-engine",
                "intent_analysis": {
                    "type": intent.intent_type.value,
                    "confidence": intent.confidence,
                    "entities": intent.entities,
                    "parameters": intent.parameters
                },
                "oracle_analysis": {
                    "code_length": len(oracle_sql),
                    "has_plsql": "DECLARE" in oracle_sql.upper() or "BEGIN" in oracle_sql.upper(),
                    "has_bulk_operations": "BULK COLLECT" in oracle_sql.upper() or "FORALL" in oracle_sql.upper(),
                    "estimated_complexity": "high" if len(oracle_sql) > 1000 else "medium" if len(oracle_sql) > 300 else "low"
                }
            }
        }
        
        if output_format == "yaml":
            import yaml
            return yaml.dump(mcp_config, default_flow_style=False)
        else:
            import json
            return json.dumps(mcp_config, indent=2)'''
            
            # Insert before the last class definition or at the end of SakthiEngine class
            if "class SakthiEngine:" in content:
                # Find the end of SakthiEngine class
                lines = content.split('\n')
                insert_pos = -1
                class_started = False
                indent_level = 0
                
                for i, line in enumerate(lines):
                    if "class SakthiEngine:" in line:
                        class_started = True
                        continue
                    
                    if class_started:
                        if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                            # Found next class or function, insert before this
                            insert_pos = i
                            break
                
                if insert_pos == -1:
                    insert_pos = len(lines)
                
                lines.insert(insert_pos, enhanced_method)
                content = '\n'.join(lines)
            else:
                content += enhanced_method
        
        # Write updated content
        with open(core_py_path, 'w') as f:
            f.write(content)
        
        logger.info("✔️ Updated core.py with enhanced MCP method")
    
    def _create_test_files(self):
        """Create test files for the dynamic MCP system"""
        
        # Create test Oracle SQL files
        test_files = {
            "test_bulk_collect.sql": """
DECLARE
  c_limit CONSTANT PLS_INTEGER := 100;
  TYPE emp_array IS TABLE OF employees%ROWTYPE;
  v_employees emp_array;
  
  CURSOR emp_cursor IS
    SELECT * FROM employees WHERE department_id = 10;
BEGIN
  OPEN emp_cursor;
  LOOP
    FETCH emp_cursor BULK COLLECT INTO v_employees LIMIT c_limit;
    EXIT WHEN v_employees.COUNT = 0;
    
    FORALL i IN 1..v_employees.COUNT
      UPDATE employees 
      SET salary = salary * 1.1 
      WHERE employee_id = v_employees(i).employee_id;
  END LOOP;
  CLOSE emp_cursor;
END;
""",
            
            "test_simple_procedure.sql": """
CREATE OR REPLACE PROCEDURE update_salary(p_emp_id NUMBER, p_percentage NUMBER) AS
BEGIN
  UPDATE employees 
  SET salary = salary * (1 + p_percentage/100)
  WHERE employee_id = p_emp_id;
  
  IF SQL%ROWCOUNT = 0 THEN
    RAISE_APPLICATION_ERROR(-20001, 'Employee not found');
  END IF;
  
  COMMIT;
END;
""",
            
            "test_complex_etl.sql": """
DECLARE
  TYPE ref_cursor IS REF CURSOR;
  c_data ref_cursor;
  v_batch_size CONSTANT NUMBER := 1000;
  
  TYPE t_staging_data IS TABLE OF staging_table%ROWTYPE;
  v_staging_data t_staging_data;
  
  v_error_count NUMBER := 0;
BEGIN
  OPEN c_data FOR
    SELECT * FROM source_table 
    WHERE created_date >= TRUNC(SYSDATE) - 1;
  
  LOOP
    FETCH c_data BULK COLLECT INTO v_staging_data LIMIT v_batch_size;
    EXIT WHEN v_staging_data.COUNT = 0;
    
    BEGIN
      FORALL i IN 1..v_staging_data.COUNT SAVE EXCEPTIONS
        INSERT INTO target_table VALUES v_staging_data(i);
        
    EXCEPTION
      WHEN OTHERS THEN
        v_error_count := v_error_count + SQL%BULK_EXCEPTIONS.COUNT;
        FOR j IN 1..SQL%BULK_EXCEPTIONS.COUNT LOOP
          INSERT INTO error_log VALUES (
            SQL%BULK_EXCEPTIONS(j).ERROR_INDEX,
            SQL%BULK_EXCEPTIONS(j).ERROR_CODE,
            SQLERRM(-SQL%BULK_EXCEPTIONS(j).ERROR_CODE),
            SYSTIMESTAMP
          );
        END LOOP;
    END;
  END LOOP;
  
  CLOSE c_data;
  DBMS_OUTPUT.PUT_LINE('Processing completed. Errors: ' || v_error_count);
END;
"""
        }
        
        test_dir = self.platform_root / "mcp_tests"
        for filename, content in test_files.items():
            test_file = test_dir / filename
            with open(test_file, 'w') as f:
                f.write(content.strip())
            logger.info(f"✔️ Created test file: {filename}")
        
        # Create test script
        test_script = '''#!/usr/bin/env python3
"""
Test script for Dynamic MCP Generation
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from mcp_engine import DynamicMCPEngine, SakthiMCPIntegration
from core import SakthiEngine

async def test_dynamic_mcp():
    """Test dynamic MCP generation with sample files"""
    
    print("🧪 Testing Dynamic MCP Generation")
    print("=" * 50)
    
    # Initialize components
    sakthi_engine = SakthiEngine()
    mcp_integration = SakthiMCPIntegration(sakthi_engine)
    
    # Test files
    test_files = [
        "test_bulk_collect.sql",
        "test_simple_procedure.sql", 
        "test_complex_etl.sql"
    ]
    
    for filename in test_files:
        test_file = Path(__file__).parent / filename
        
        if not test_file.exists():
            print(f"❌ Test file not found: {filename}")
            continue
        
        print(f"\\n📁 Testing: {filename}")
        print("-" * 30)
        
        # Read test file
        with open(test_file, 'r') as f:
            oracle_sql = f.read()
        
        # Process with dynamic MCP
        result = await mcp_integration.process_oracle_file_to_mcp(
            oracle_sql, filename, include_llm_analysis=True
        )
        
        if result["success"]:
            print(f"✅ Success!")
            print(f"   LLM Analysis: {result['llm_analysis']['description'] if result['llm_analysis'] else 'No analysis'}")
            print(f"   Validation Score: {result['validation']['score'] if result['validation'] else 'N/A'}%")
            print(f"   Template: {result['download_path'] if result['download_path'] else 'Not created'}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    print("\\n✅ Dynamic MCP testing completed!")

if __name__ == "__main__":
    asyncio.run(test_dynamic_mcp())
'''
        
        test_script_file = test_dir / "test_dynamic_mcp.py"
        with open(test_script_file, 'w') as f:
            f.write(test_script)
        
        # Make executable
        os.chmod(test_script_file, 0o755)
        logger.info("✔️ Created test script: test_dynamic_mcp.py")
    
    def _setup_requirements(self):
        """Setup additional requirements for dynamic MCP"""
        
        additional_requirements = [
            "jinja2>=3.0.0",
            "asyncio",
            "aiofiles", 
            "python-multipart"
        ]
        
        requirements_file = self.platform_root / "requirements_mcp.txt"
        with open(requirements_file, 'w') as f:
            f.write("# Additional requirements for Dynamic MCP System\\n")
            for req in additional_requirements:
                f.write(f"{req}\\n")
        
        logger.info("✔️ Created requirements_mcp.txt")
        
        # Create install script
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
        
        install_script_file = self.platform_root / "install_mcp_requirements.py"
        with open(install_script_file, 'w') as f:
            f.write(install_script)
        
        os.chmod(install_script_file, 0o755)
        logger.info("✔️ Created install script: install_mcp_requirements.py")
    
    def _verify_installation(self):
        """Verify the dynamic MCP installation"""
        
        required_files = [
            "mcp_templates/oracle_bigquery_config.json",
            "mcp_engine.py",
            "mcp_visual_interface.py",
            "mcp_tests/test_dynamic_mcp.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.platform_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            raise Exception(f"Missing required files: {missing_files}")
        
        logger.info("✅ Installation verification passed")
    
    def _print_usage_instructions(self):
        """Print usage instructions"""
        
        instructions = """
🎉 Dynamic MCP System Setup Complete!

📁 Files Created:
├── mcp_templates/
│   ├── oracle_bigquery_config.json    # Main template configuration
│   ├── oracle_analysis_prompts.json   # LLM analysis prompts
│   ├── bigquery_generation_prompts.json # BigQuery generation prompts
│   └── validation_config.json         # Validation rules
├── mcp_engine.py                      # Dynamic MCP engine
├── mcp_visual_interface.py            # Visual interface integration
├── mcp_tests/                         # Test files and scripts
└── generated_mcp/                     # Output directory

🚀 Quick Start:

1. Install additional requirements:
   python install_mcp_requirements.py

2. Test the system:
   cd mcp_tests
   python test_dynamic_mcp.py

3. Use in your Sakthi platform:
   - Upload Oracle SQL files
   - Get dynamic MCP configs with LLM analysis
   - Download templates with validation

🎯 Features:
✅ LLM-powered Oracle analysis
✅ Dynamic MCP generation with Jinja2 templates
✅ Visual flyout cart experience
✅ Validation and quality scoring
✅ Downloadable templates
✅ Retry mechanisms and error handling

📖 Next Steps:
- Customize templates in mcp_templates/
- Add your own validation rules
- Extend visual interface components
- Create custom prompt templates

🔧 Integration:
Your backend/main.py has been updated to use the dynamic MCP system.
Restart your backend to use the new features!
"""
        
        print(instructions)

def main():
    """Main setup function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Dynamic MCP System")
    parser.add_argument("--platform-root", default=".", help="Platform root directory")
    parser.add_argument("--force", action="store_true", help="Force setup even if files exist")
    
    args = parser.parse_args()
    
    setup = DynamicMCPSetup(args.platform_root)
    setup.run_complete_setup()

if __name__ == "__main__":
    main()