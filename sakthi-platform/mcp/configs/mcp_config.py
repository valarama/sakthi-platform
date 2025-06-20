# mcp_config.py
"""
MCP Configuration and Integration Setup for Sakthi Platform
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MCPConfig:
    """MCP Configuration Manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(os.getcwd(), "config", "mcp_config.yaml")
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load MCP configuration"""
        
        default_config = {
            "mcp": {
                "server": {
                    "name": "sakthi-mcp-server",
                    "version": "1.0.0",
                    "host": "localhost",
                    "port": 8080
                },
                "capabilities": {
                    "resources": True,
                    "tools": True,
                    "prompts": True,
                    "logging": True
                },
                "output_formats": ["json", "yaml"],
                "base_uri": "sakthi://",
                "max_resources": 1000,
                "timeout": 30
            },
            "sakthi": {
                "core": {
                    "confidence_threshold": 0.7,
                    "max_entities": 50,
                    "supported_intents": [
                        "data_extraction",
                        "schema_mapping", 
                        "transformation",
                        "migration",
                        "analysis"
                    ]
                },
                "genai": {
                    "model": "gpt-4",
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "timeout": 60
                }
            },
            "integration": {
                "auto_generate_mcp": True,
                "validate_output": True,
                "cache_results": True,
                "batch_processing": True
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/mcp_integration.log"
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    # Merge with defaults
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Could not load config from {self.config_path}: {e}")
        
        return default_config
    
    def save_config(self):
        """Save current configuration"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, key_path: str, value):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value

# MCP Integration Script
class MCPIntegrationSetup:
    """Setup MCP integration with existing Sakthi platform"""
    
    def __init__(self, platform_root: str):
        self.platform_root = Path(platform_root)
        self.config = MCPConfig()
        
    def setup_integration(self):
        """Setup complete MCP integration"""
        
        print("Setting up MCP integration for Sakthi platform...")
        
        # 1. Create MCP directory structure
        self._create_mcp_directories()
        
        # 2. Copy MCP generator to appropriate location
        self._setup_mcp_generator()
        
        # 3. Create integration scripts
        self._create_integration_scripts()
        
        # 4. Update existing modules
        self._update_existing_modules()
        
        # 5. Create example usage scripts
        self._create_examples()
        
        # 6. Generate sample MCP configurations
        self._generate_sample_configs()
        
        print("✅ MCP integration setup complete!")
        
    def _create_mcp_directories(self):
        """Create MCP directory structure"""
        
        mcp_dirs = [
            "mcp",
            "mcp/generators",
            "mcp/configs", 
            "mcp/examples",
            "mcp/schemas",
            "mcp/tools",
            "mcp/prompts",
            "mcp/resources"
        ]
        
        for dir_name in mcp_dirs:
            dir_path = self.platform_root / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py for Python packages
            if not str(dir_path).endswith(('configs', 'examples', 'schemas')):
                (dir_path / "__init__.py").touch()
            
            print(f"✔️ Created directory: {dir_path}")
    
    def _setup_mcp_generator(self):
        """Setup MCP generator in the platform"""
        
        mcp_generator_content = '''# Placeholder for MCP generator
# Copy the mcp_generator.py content here
from .mcp_generator import SakthiMCPGenerator, SakthiMCPIntegration
'''
        
        init_file = self.platform_root / "mcp" / "__init__.py"
        with open(init_file, 'w') as f:
            f.write(mcp_generator_content)
        
        print("✔️ MCP generator setup complete")
    
    def _create_integration_scripts(self):
        """Create integration scripts"""
        
        # Main integration script
        integration_script = '''#!/usr/bin/env python3
"""
Sakthi-MCP Integration Script
Usage: python integrate_mcp.py [options]
"""

import sys
import os
import argparse
import asyncio
from pathlib import Path

# Add platform root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import SakthiEngine
from genai_modeling_agent.agent_system import GenAIModelingAgent
from mcp.mcp_generator import SakthiMCPIntegration
from mcp_config import MCPConfig

async def main():
    parser = argparse.ArgumentParser(description='Sakthi-MCP Integration')
    parser.add_argument('--input', '-i', required=True, help='Input text or file')
    parser.add_argument('--format', '-f', choices=['json', 'yaml'], default='json', help='Output format')
    parser.add_argument('--type', '-t', choices=['nlp', 'schema', 'modeling'], default='nlp', help='Processing type')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--config', '-c', help='Config file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = MCPConfig(args.config)
    
    # Initialize Sakthi components
    sakthi_engine = SakthiEngine()
    
    genai_config = {
        'llm_config': config.get('sakthi.genai')
    }
    genai_agent = GenAIModelingAgent(genai_config)
    
    # Initialize MCP integration
    mcp_integration = SakthiMCPIntegration(sakthi_engine, genai_agent)
    
    # Process input
    if args.type == 'nlp':
        result = mcp_integration.process_nlp_to_mcp(args.input, args.format)
    elif args.type == 'modeling':
        # Parse input as JSON for modeling request
        import json
        request = json.loads(args.input) if args.input.startswith('{') else {'input': args.input}
        result = await mcp_integration.process_modeling_request_to_mcp(request, args.format)
    else:
        print(f"Processing type '{args.type}' not yet implemented")
        return
    
    # Output result
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
        print(f"MCP configuration saved to: {args.output}")
    else:
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        script_path = self.platform_root / "integrate_mcp.py"
        with open(script_path, 'w') as f:
            f.write(integration_script)
        
        # Make executable
        os.chmod(script_path, 0o755)
        print(f"✔️ Created integration script: {script_path}")
        
        # Create batch processing script
        batch_script = '''#!/usr/bin/env python3
"""
Batch MCP Generation Script
Process multiple inputs and generate MCP configurations
"""

import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any

from mcp_config import MCPConfig
from core import SakthiEngine
from genai_modeling_agent.agent_system import GenAIModelingAgent
from mcp.mcp_generator import SakthiMCPIntegration

class BatchMCPProcessor:
    def __init__(self, config_path: str = None):
        self.config = MCPConfig(config_path)
        self.sakthi_engine = SakthiEngine()
        
        genai_config = {'llm_config': self.config.get('sakthi.genai')}
        self.genai_agent = GenAIModelingAgent(genai_config)
        
        self.mcp_integration = SakthiMCPIntegration(self.sakthi_engine, self.genai_agent)
    
    async def process_batch(self, inputs: List[Dict[str, Any]], output_dir: str):
        """Process batch of inputs"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, input_item in enumerate(inputs):
            try:
                input_text = input_item.get('text', '')
                input_type = input_item.get('type', 'nlp')
                output_format = input_item.get('format', 'json')
                
                print(f"Processing item {i+1}/{len(inputs)}: {input_text[:50]}...")
                
                if input_type == 'nlp':
                    result = self.mcp_integration.process_nlp_to_mcp(input_text, output_format)
                elif input_type == 'modeling':
                    result = await self.mcp_integration.process_modeling_request_to_mcp(input_item, output_format)
                
                # Save result
                filename = f"mcp_config_{i+1}.{output_format}"
                output_file = output_path / filename
                
                with open(output_file, 'w') as f:
                    f.write(result)
                
                results.append({
                    'input': input_item,
                    'output_file': str(output_file),
                    'success': True
                })
                
                print(f"✔️ Saved: {output_file}")
                
            except Exception as e:
                print(f"❌ Error processing item {i+1}: {e}")
                results.append({
                    'input': input_item,
                    'error': str(e),
                    'success': False
                })
        
        # Save batch summary
        summary_file = output_path / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\n✅ Batch processing complete. Summary: {summary_file}")
        
        return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch MCP Processing')
    parser.add_argument('--input-file', '-i', required=True, help='JSON file with batch inputs')
    parser.add_argument('--output-dir', '-o', required=True, help='Output directory')
    parser.add_argument('--config', '-c', help='Config file path')
    
    args = parser.parse_args()
    
    # Load inputs
    with open(args.input_file, 'r') as f:
        inputs = json.load(f)
    
    # Process batch
    processor = BatchMCPProcessor(args.config)
    asyncio.run(processor.process_batch(inputs, args.output_dir))
'''
        
        batch_script_path = self.platform_root / "batch_mcp.py"
        with open(batch_script_path, 'w') as f:
            f.write(batch_script)
        
        os.chmod(batch_script_path, 0o755)
        print(f"✔️ Created batch processing script: {batch_script_path}")
    
    def _update_existing_modules(self):
        """Update existing modules to include MCP support"""
        
        # Update core.py to include MCP export
        core_addition = '''
    def export_to_mcp(self, output_format: OutputFormat = OutputFormat.JSON) -> str:
        """Export Sakthi output to MCP format"""
        from mcp.mcp_generator import SakthiMCPGenerator
        
        mcp_generator = SakthiMCPGenerator()
        
        # Create a basic intent from the current engine state
        intent = SakthiIntent(
            intent_type=IntentType.ANALYSIS,
            confidence=0.8,
            entities={},
            source_format="unknown",
            target_format=output_format.value,
            parameters={}
        )
        
        mcp_server = mcp_generator.generate_mcp_from_intent(intent)
        
        if output_format == OutputFormat.YAML:
            return mcp_generator.export_mcp_yaml(mcp_server)
        else:
            return mcp_generator.export_mcp_json(mcp_server)
'''
        
        # Create a patch file for core.py
        patch_file = self.platform_root / "mcp" / "core_mcp_patch.py"
        with open(patch_file, 'w') as f:
            f.write(f"# Add this method to the SakthiEngine class in core.py\\n{core_addition}")
        
        print(f"✔️ Created core.py patch: {patch_file}")
        
        # Update agent_system.py integration
        agent_addition = '''
    async def export_mcpp_to_mcp(self, mcpp_schema: MCPPSchema, output_format: str = "json") -> str:
        """Export MCPP schema to MCP format"""
        from mcp.mcp_generator import SakthiMCPGenerator
        
        mcp_generator = SakthiMCPGenerator()
        mcp_server = mcp_generator.generate_mcp_from_schema(mcpp_schema)
        
        if output_format.lower() == "yaml":
            return mcp_generator.export_mcp_yaml(mcp_server)
        else:
            return mcp_generator.export_mcp_json(mcp_server)
'''
        
        agent_patch_file = self.platform_root / "mcp" / "agent_mcp_patch.py"
        with open(agent_patch_file, 'w') as f:
            f.write(f"# Add this method to the GenAIModelingAgent class\\n{agent_addition}")
        
        print(f"✔️ Created agent_system.py patch: {agent_patch_file}")
    
    def _create_examples(self):
        """Create example usage scripts"""
        
        examples = {
            "basic_nlp_to_mcp.py": '''#!/usr/bin/env python3
"""
Basic NLP to MCP conversion example
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core import SakthiEngine
from mcp.mcp_generator import SakthiMCPGenerator

def main():
    # Initialize components
    sakthi_engine = SakthiEngine()
    mcp_generator = SakthiMCPGenerator()
    
    # Example inputs
    examples = [
        "Extract customer data from PostgreSQL customers table",
        "Map Oracle HR schema to BigQuery with data type optimization",
        "Transform CSV sales data to JSON format with validation",
        "Migrate inventory database from MySQL to Snowflake"
    ]
    
    print("=== NLP to MCP Conversion Examples ===\\n")
    
    for i, example_text in enumerate(examples, 1):
        print(f"Example {i}: {example_text}")
        print("-" * 60)
        
        # Parse with Sakthi
        intent = sakthi_engine.parser.parse(example_text)
        print(f"Detected Intent: {intent.intent_type.value}")
        print(f"Confidence: {intent.confidence:.2f}")
        print(f"Entities: {intent.entities}")
        
        # Generate MCP
        mcp_server = mcp_generator.generate_mcp_from_intent(intent)
        
        # Show summary
        print(f"MCP Resources: {len(mcp_server.resources)}")
        print(f"MCP Tools: {len(mcp_server.tools)}")
        print(f"MCP Prompts: {len(mcp_server.prompts)}")
        
        # Export sample
        mcp_json = mcp_generator.export_mcp_json(mcp_server)
        
        # Save to file
        output_file = Path(f"mcp_example_{i}.json")
        with open(output_file, 'w') as f:
            f.write(mcp_json)
        
        print(f"✔️ Saved MCP config: {output_file}")
        print("\\n" + "="*70 + "\\n")

if __name__ == "__main__":
    main()
''',
            
            "schema_to_mcp.py": '''#!/usr/bin/env python3
"""
Schema to MCP conversion example
"""

import sys
import asyncio
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from genai_modeling_agent.agent_system import GenAIModelingAgent, MCPPSchema
from mcp.mcp_generator import SakthiMCPGenerator
from datetime import datetime

async def main():
    # Create sample MCPP schema
    sample_schema = MCPPSchema(
        schema_id="sample_hr_migration",
        source_system="oracle",
        target_system="bigquery",
        tables=[
            {
                "source_table": "employees",
                "target_table": "hr_employees",
                "column_mappings": [
                    {"source": "emp_id", "target": "employee_id", "transformation": "direct"},
                    {"source": "first_name", "target": "first_name", "transformation": "direct"},
                    {"source": "last_name", "target": "last_name", "transformation": "direct"},
                    {"source": "hire_date", "target": "hire_date", "transformation": "date_format"}
                ]
            },
            {
                "source_table": "departments",
                "target_table": "hr_departments", 
                "column_mappings": [
                    {"source": "dept_id", "target": "department_id", "transformation": "direct"},
                    {"source": "dept_name", "target": "department_name", "transformation": "direct"}
                ]
            }
        ],
        relationships=[
            {
                "type": "foreign_key",
                "source_table": "employees",
                "source_column": "dept_id",
                "target_table": "departments",
                "target_column": "dept_id"
            }
        ],
        transformations=[
            {
                "type": "data_type_conversion",
                "description": "Convert Oracle DATE to BigQuery TIMESTAMP"
            },
            {
                "type": "table_rename",
                "description": "Add hr_ prefix to all tables"
            }
        ],
        metadata={
            "complexity_score": 0.4,
            "estimated_migration_time": "2 hours",
            "data_volume": "medium"
        }
    )
    
    # Generate MCP from schema
    mcp_generator = SakthiMCPGenerator()
    mcp_server = mcp_generator.generate_mcp_from_schema(sample_schema)
    
    print("=== Schema to MCP Conversion Example ===\\n")
    print(f"Schema ID: {sample_schema.schema_id}")
    print(f"Source: {sample_schema.source_system} -> Target: {sample_schema.target_system}")
    print(f"Tables: {len(sample_schema.tables)}")
    print(f"Relationships: {len(sample_schema.relationships)}")
    print(f"Transformations: {len(sample_schema.transformations)}")
    print()
    
    # Export MCP configurations
    mcp_json = mcp_generator.export_mcp_json(mcp_server)
    mcp_yaml = mcp_generator.export_mcp_yaml(mcp_server)
    
    # Save outputs
    with open("schema_mcp_config.json", "w") as f:
        f.write(mcp_json)
    
    with open("schema_mcp_config.yaml", "w") as f:
        f.write(mcp_yaml)
    
    print("✔️ Generated MCP configurations:")
    print("  - schema_mcp_config.json")
    print("  - schema_mcp_config.yaml")
    print()
    
    # Show MCP summary
    print("MCP Server Summary:")
    print(f"  Name: {mcp_server.name}")
    print(f"  Version: {mcp_server.version}")
    print(f"  Resources: {len(mcp_server.resources)}")
    print(f"  Tools: {len(mcp_server.tools)}")
    print(f"  Prompts: {len(mcp_server.prompts)}")

if __name__ == "__main__":
    asyncio.run(main())
'''
        }
        
        examples_dir = self.platform_root / "mcp" / "examples"
        
        for filename, content in examples.items():
            example_file = examples_dir / filename
            with open(example_file, 'w') as f:
                f.write(content)
            
            os.chmod(example_file, 0o755)
            print(f"✔️ Created example: {example_file}")
    
    def _generate_sample_configs(self):
        """Generate sample MCP configurations"""
        
        configs_dir = self.platform_root / "mcp" / "configs"
        
        # Sample batch input file
        batch_input = [
            {
                "text": "Extract sales data from PostgreSQL for Q4 analysis",
                "type": "nlp",
                "format": "json"
            },
            {
                "text": "Map Oracle HR schema to Snowflake with performance optimization",
                "type": "nlp", 
                "format": "yaml"
            },
            {
                "text": "Transform Excel financial reports to structured JSON",
                "type": "nlp",
                "format": "json"
            }
        ]
        
        batch_file = configs_dir / "sample_batch_input.json"
        with open(batch_file, 'w') as f:
            json.dump(batch_input, f, indent=2)
        
        print(f"✔️ Created sample batch input: {batch_file}")
        
        # Sample MCP server config template
        mcp_template = {
            "mcpVersion": "2024-11-05",
            "server": {
                "name": "sakthi-mcp-server",
                "version": "1.0.0"
            },
            "capabilities": {
                "resources": {"subscribe": True, "listChanged": True},
                "tools": {"listChanged": True},
                "prompts": {"listChanged": True},
                "logging": {"level": "info"}
            },
            "resources": [],
            "tools": [],
            "prompts": [],
            "metadata": {
                "generated_by": "sakthi-platform",
                "template": True
            }
        }
        
        template_file = configs_dir / "mcp_template.json"
        with open(template_file, 'w') as f:
            json.dump(mcp_template, f, indent=2)
        
        print(f"✔️ Created MCP template: {template_file}")

# Quick setup function
def quick_setup():
    """Quick setup for MCP integration"""
    
    current_dir = os.getcwd()
    print(f"Setting up MCP integration in: {current_dir}")
    
    setup = MCPIntegrationSetup(current_dir)
    setup.setup_integration()
    
    print("\\n🎉 MCP Integration Setup Complete!")
    print("\\nNext steps:")
    print("1. Copy mcp_generator.py to mcp/generators/")
    print("2. Run: python mcp/examples/basic_nlp_to_mcp.py")
    print("3. Test: python integrate_mcp.py -i 'Extract data from users table' -f json")
    print("4. Batch process: python batch_mcp.py -i mcp/configs/sample_batch_input.json -o output/")

if __name__ == "__main__":
    quick_setup()