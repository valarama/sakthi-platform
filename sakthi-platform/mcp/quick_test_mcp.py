#!/usr/bin/env python3
"""
Quick MCP Test Script
Test the NLP to MCP conversion with your existing Sakthi platform
"""

import os
import sys
import json
from pathlib import Path

# Add your platform paths
platform_root = Path.cwd()
sys.path.insert(0, str(platform_root))

def test_basic_mcp_generation():
    """Test basic MCP generation without complex dependencies"""
    
    print("=== Quick MCP Generation Test ===\n")
    
    # Mock Sakthi components for testing
    class MockSakthiIntent:
        def __init__(self, text):
            self.intent_type_value = self._detect_intent(text)
            self.confidence = 0.85
            self.entities = self._extract_entities(text)
            self.source_format = self._detect_source(text)
            self.target_format = self._detect_target(text)
            self.parameters = {}
        
        def _detect_intent(self, text):
            text_lower = text.lower()
            if any(word in text_lower for word in ['extract', 'get', 'retrieve']):
                return 'data_extraction'
            elif any(word in text_lower for word in ['map', 'convert', 'migrate']):
                return 'schema_mapping'
            elif any(word in text_lower for word in ['transform', 'change']):
                return 'transformation'
            else:
                return 'analysis'
        
        def _extract_entities(self, text):
            entities = {}
            
            # Simple entity extraction
            databases = ['oracle', 'mysql', 'postgresql', 'bigquery', 'snowflake']
            formats = ['csv', 'json', 'xml', 'xlsx', 'pdf']
            
            for db in databases:
                if db in text.lower():
                    entities.setdefault('database', []).append(db)
            
            for fmt in formats:
                if fmt in text.lower():
                    entities.setdefault('file_format', []).append(fmt)
            
            return entities
        
        def _detect_source(self, text):
            if 'oracle' in text.lower():
                return 'oracle'
            elif 'mysql' in text.lower():
                return 'mysql'
            elif 'csv' in text.lower():
                return 'csv'
            return 'unknown'
        
        def _detect_target(self, text):
            if 'bigquery' in text.lower():
                return 'bigquery'
            elif 'json' in text.lower():
                return 'json'
            elif 'snowflake' in text.lower():
                return 'snowflake'
            return 'json'
    
    def generate_mcp_config(intent):
        """Generate basic MCP configuration"""
        
        mcp_config = {
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
            "resources": [
                {
                    "uri": f"sakthi://intent/{intent.intent_type_value}",
                    "name": f"intent_{intent.intent_type_value}",
                    "description": f"Sakthi intent for {intent.intent_type_value}",
                    "mimeType": "application/json",
                    "annotations": {
                        "confidence": intent.confidence,
                        "entities": intent.entities,
                        "source_format": intent.source_format,
                        "target_format": intent.target_format
                    }
                }
            ],
            "tools": [
                {
                    "name": "process_intent",
                    "description": f"Process {intent.intent_type_value} intent",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "input_text": {"type": "string", "description": "Natural language input"},
                            "context": {"type": "object", "description": "Processing context"},
                            "output_format": {"type": "string", "enum": ["json", "yaml", "sql"]}
                        },
                        "required": ["input_text"]
                    }
                }
            ],
            "prompts": [
                {
                    "name": "sakthi_processor",
                    "description": f"Process {intent.intent_type_value} with Sakthi",
                    "arguments": [
                        {"name": "input_text", "description": "Input text", "required": True},
                        {"name": "format", "description": "Output format", "required": False}
                    ],
                    "template": f"Process this {intent.intent_type_value} request: {{input_text}}"
                }
            ],
            "metadata": {
                "generated_by": "sakthi-platform",
                "intent_type": intent.intent_type_value,
                "confidence": intent.confidence,
                "source_format": intent.source_format,
                "target_format": intent.target_format,
                "entities": intent.entities
            }
        }
        
        return mcp_config
    
    # Test cases
    test_cases = [
        "Extract customer data from Oracle database to BigQuery",
        "Convert MySQL user schema to Snowflake format",
        "Transform CSV sales data to JSON with validation",
        "Map PostgreSQL inventory tables to BigQuery analytics"
    ]
    
    results = []
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"Test {i}: {test_input}")
        print("-" * 60)
        
        # Create mock intent
        intent = MockSakthiIntent(test_input)
        
        # Generate MCP config
        mcp_config = generate_mcp_config(intent)
        
        # Show summary
        print(f"Intent Type: {intent.intent_type_value}")
        print(f"Confidence: {intent.confidence}")
        print(f"Source: {intent.source_format} -> Target: {intent.target_format}")
        print(f"Entities: {intent.entities}")
        print(f"Resources: {len(mcp_config['resources'])}")
        print(f"Tools: {len(mcp_config['tools'])}")
        print(f"Prompts: {len(mcp_config['prompts'])}")
        
        # Save output
        output_file = f"test_mcp_config_{i}.json"
        with open(output_file, 'w') as f:
            json.dump(mcp_config, f, indent=2)
        
        print(f"✔️ Generated: {output_file}")
        print()
        
        results.append({
            'test_case': test_input,
            'intent_type': intent.intent_type_value,
            'confidence': intent.confidence,
            'output_file': output_file
        })
    
    # Create summary
    summary = {
        'test_summary': {
            'total_tests': len(test_cases),
            'successful_generations': len(results),
            'average_confidence': sum(r['confidence'] for r in results) / len(results)
        },
        'results': results
    }
    
    with open('mcp_test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("="*70)
    print("✅ MCP Generation Test Complete!")
    print(f"Generated {len(results)} MCP configurations")
    print(f"Average confidence: {summary['test_summary']['average_confidence']:.2f}")
    print("Files created:")
    for result in results:
        print(f"  - {result['output_file']}")
    print("  - mcp_test_summary.json")
    print()
    print("Next steps:")
    print("1. Copy the mcp_generator.py code to your platform")
    print("2. Integrate with your existing core.py and agent_system.py")
    print("3. Run the full integration with: python mcp_config.py")

def check_platform_structure():
    """Check if we're in the right directory"""
    import os
    expected_files = [
        'core.py',
        os.path.join('genai-modeling-agent', 'agent_system.py'),
        os.path.join('document-processor', 'processor.py')
    ]
    missing_files = []
    print("Checking platform structure...")
    for file in expected_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✔️ Found: {file}")
    if missing_files:
        print(f"⚠️  Missing files: {missing_files}")
        print("Make sure you're running this from your sakthi-platform root directory")
        return False
    print("✅ Platform structure looks good!")
    return True

def main():
    """Main test function"""
    
    print("🚀 Sakthi Platform MCP Integration Test\n")
    
    # Check platform structure
    if not check_platform_structure():
        print("\n❌ Platform structure check failed")
        print("Please run this script from your sakthi-platform directory")
        return
    
    print()
    
    # Run the test
    test_basic_mcp_generation()

if __name__ == "__main__":
    main()