# mcp_generator.py
"""
MCP (Model Context Protocol) Language Generator for Sakthi Platform
Converts Sakthi NLP intents and schemas into MCP format
"""

import json
import yaml
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import uuid
import logging

# Import from your existing modules
from core import SakthiIntent, SakthiOutput, OutputFormat, IntentType
from agent_system import MCPPSchema, ProcessingContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPResourceType(Enum):
    """MCP Resource Types"""
    SCHEMA = "schema"
    TABLE = "table"
    COLUMN = "column"
    RELATIONSHIP = "relationship"
    TRANSFORMATION = "transformation"
    TOOL = "tool"
    PROMPT = "prompt"

class MCPCapability(Enum):
    """MCP Capabilities"""
    RESOURCES = "resources"
    TOOLS = "tools"
    PROMPTS = "prompts"
    LOGGING = "logging"

@dataclass
class MCPResource:
    """MCP Resource Definition"""
    uri: str
    name: str
    description: str
    mimeType: str
    annotations: Optional[Dict[str, Any]] = None

@dataclass
class MCPTool:
    """MCP Tool Definition"""
    name: str
    description: str
    inputSchema: Dict[str, Any]
    outputSchema: Optional[Dict[str, Any]] = None

@dataclass
class MCPPrompt:
    """MCP Prompt Definition"""
    name: str
    description: str
    arguments: List[Dict[str, Any]]
    template: str

@dataclass
class MCPServer:
    """MCP Server Configuration"""
    name: str
    version: str
    capabilities: Dict[str, Any]
    resources: List[MCPResource]
    tools: List[MCPTool]
    prompts: List[MCPPrompt]
    metadata: Dict[str, Any]

class SakthiMCPGenerator:
    """Generates MCP language format from Sakthi NLP components"""
    
    def __init__(self):
        self.server_name = "sakthi-mcp-server"
        self.server_version = "1.0.0"
        self.base_uri = "sakthi://"
    
    def generate_mcp_from_intent(self, intent: SakthiIntent) -> MCPServer:
        """Generate MCP server configuration from Sakthi intent"""
        
        logger.info(f"Generating MCP for intent: {intent.intent_type.value}")
        
        # Generate capabilities based on intent
        capabilities = self._generate_capabilities(intent)
        
        # Generate resources
        resources = self._generate_resources(intent)
        
        # Generate tools
        tools = self._generate_tools(intent)
        
        # Generate prompts
        prompts = self._generate_prompts(intent)
        
        # Generate metadata
        metadata = {
            "generated_from": "sakthi_intent",
            "intent_type": intent.intent_type.value,
            "confidence": intent.confidence,
            "source_format": intent.source_format,
            "target_format": intent.target_format,
            "generation_time": datetime.now().isoformat(),
            "sakthi_version": "1.0.0"
        }
        
        return MCPServer(
            name=self.server_name,
            version=self.server_version,
            capabilities=capabilities,
            resources=resources,
            tools=tools,
            prompts=prompts,
            metadata=metadata
        )
    
    def generate_mcp_from_schema(self, mcpp_schema: MCPPSchema) -> MCPServer:
        """Generate MCP server configuration from MCPP schema"""
        
        logger.info(f"Generating MCP for schema: {mcpp_schema.schema_id}")
        
        # Generate capabilities for schema operations
        capabilities = {
            "resources": {"subscribe": True, "listChanged": True},
            "tools": {"listChanged": True},
            "prompts": {"listChanged": True},
            "logging": {"level": "info"}
        }
        
        # Generate resources from schema
        resources = self._generate_schema_resources(mcpp_schema)
        
        # Generate schema-specific tools
        tools = self._generate_schema_tools(mcpp_schema)
        
        # Generate schema-specific prompts
        prompts = self._generate_schema_prompts(mcpp_schema)
        
        # Generate metadata
        metadata = {
            "generated_from": "mcpp_schema",
            "schema_id": mcpp_schema.schema_id,
            "source_system": mcpp_schema.source_system,
            "target_system": mcpp_schema.target_system,
            "table_count": len(mcpp_schema.tables),
            "relationship_count": len(mcpp_schema.relationships),
            "transformation_count": len(mcpp_schema.transformations),
            "generation_time": datetime.now().isoformat()
        }
        
        return MCPServer(
            name=f"sakthi-schema-{mcpp_schema.schema_id[:8]}",
            version=self.server_version,
            capabilities=capabilities,
            resources=resources,
            tools=tools,
            prompts=prompts,
            metadata=metadata
        )
    
    def _generate_capabilities(self, intent: SakthiIntent) -> Dict[str, Any]:
        """Generate MCP capabilities based on intent"""
        
        base_capabilities = {
            "resources": {"subscribe": True, "listChanged": True},
            "tools": {"listChanged": True},
            "prompts": {"listChanged": True},
            "logging": {"level": "info"}
        }
        
        # Add specific capabilities based on intent type
        if intent.intent_type == IntentType.DATA_EXTRACTION:
            base_capabilities["tools"]["data_extraction"] = True
            base_capabilities["resources"]["data_sources"] = True
        
        elif intent.intent_type == IntentType.SCHEMA_MAPPING:
            base_capabilities["tools"]["schema_mapping"] = True
            base_capabilities["resources"]["schema_definitions"] = True
        
        elif intent.intent_type == IntentType.TRANSFORMATION:
            base_capabilities["tools"]["data_transformation"] = True
            base_capabilities["resources"]["transformation_rules"] = True
        
        elif intent.intent_type == IntentType.MIGRATION:
            base_capabilities["tools"]["data_migration"] = True
            base_capabilities["resources"]["migration_plans"] = True
        
        return base_capabilities
    
    def _generate_resources(self, intent: SakthiIntent) -> List[MCPResource]:
        """Generate MCP resources from intent"""
        
        resources = []
        
        # Base intent resource
        resources.append(MCPResource(
            uri=f"{self.base_uri}intent/{intent.intent_type.value}",
            name=f"intent_{intent.intent_type.value}",
            description=f"Sakthi intent for {intent.intent_type.value}",
            mimeType="application/json",
            annotations={
                "confidence": intent.confidence,
                "entities": intent.entities,
                "parameters": intent.parameters
            }
        ))
        
        # Entity-based resources
        for entity_type, entity_values in intent.entities.items():
            if entity_values:  # Only if entities exist
                resources.append(MCPResource(
                    uri=f"{self.base_uri}entities/{entity_type}",
                    name=f"entities_{entity_type}",
                    description=f"Extracted {entity_type} entities",
                    mimeType="application/json",
                    annotations={"values": entity_values}
                ))
        
        # Source/target format resources
        if intent.source_format != 'unknown':
            resources.append(MCPResource(
                uri=f"{self.base_uri}formats/source/{intent.source_format}",
                name=f"source_format_{intent.source_format}",
                description=f"Source format: {intent.source_format}",
                mimeType="application/json"
            ))
        
        if intent.target_format:
            resources.append(MCPResource(
                uri=f"{self.base_uri}formats/target/{intent.target_format}",
                name=f"target_format_{intent.target_format}",
                description=f"Target format: {intent.target_format}",
                mimeType="application/json"
            ))
        
        return resources
    
    def _generate_tools(self, intent: SakthiIntent) -> List[MCPTool]:
        """Generate MCP tools from intent"""
        
        tools = []
        
        # Base processing tool
        tools.append(MCPTool(
            name="process_intent",
            description=f"Process {intent.intent_type.value} intent",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_text": {"type": "string", "description": "Natural language input"},
                    "context": {"type": "object", "description": "Processing context"},
                    "output_format": {"type": "string", "enum": ["json", "yaml", "sql", "python"]}
                },
                "required": ["input_text"]
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "result": {"type": "object", "description": "Processing result"},
                    "confidence": {"type": "number", "description": "Confidence score"},
                    "metadata": {"type": "object", "description": "Processing metadata"}
                }
            }
        ))
        
        # Intent-specific tools
        if intent.intent_type == IntentType.DATA_EXTRACTION:
            tools.append(MCPTool(
                name="extract_data",
                description="Extract data from specified sources",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Data source"},
                        "fields": {"type": "array", "items": {"type": "string"}},
                        "filters": {"type": "object", "description": "Data filters"}
                    },
                    "required": ["source"]
                }
            ))
        
        elif intent.intent_type == IntentType.SCHEMA_MAPPING:
            tools.append(MCPTool(
                name="map_schema",
                description="Map schema from source to target",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source_schema": {"type": "object", "description": "Source schema"},
                        "target_schema": {"type": "object", "description": "Target schema"},
                        "mapping_rules": {"type": "object", "description": "Mapping rules"}
                    },
                    "required": ["source_schema", "target_schema"]
                }
            ))
        
        elif intent.intent_type == IntentType.TRANSFORMATION:
            tools.append(MCPTool(
                name="transform_data",
                description="Transform data according to rules",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "Input data"},
                        "transformations": {"type": "array", "description": "Transformation rules"},
                        "validation": {"type": "boolean", "description": "Validate output"}
                    },
                    "required": ["data", "transformations"]
                }
            ))
        
        return tools
    
    def _generate_prompts(self, intent: SakthiIntent) -> List[MCPPrompt]:
        """Generate MCP prompts from intent"""
        
        prompts = []
        
        # Base intent prompt
        prompts.append(MCPPrompt(
            name="sakthi_intent_processor",
            description=f"Process {intent.intent_type.value} with Sakthi language",
            arguments=[
                {"name": "input_text", "description": "Natural language input", "required": True},
                {"name": "context", "description": "Additional context", "required": False},
                {"name": "output_format", "description": "Desired output format", "required": False}
            ],
            template="""
Process the following {intent_type} request using Sakthi language understanding:

Input: {input_text}
Context: {context}
Output Format: {output_format}

Entities detected: {entities}
Parameters: {parameters}
Confidence: {confidence}

Please generate structured output in the requested format.
"""
        ))
        
        # Intent-specific prompts
        if intent.intent_type == IntentType.SCHEMA_MAPPING:
            prompts.append(MCPPrompt(
                name="schema_mapping_assistant",
                description="Assist with schema mapping tasks",
                arguments=[
                    {"name": "source_schema", "description": "Source schema definition", "required": True},
                    {"name": "target_schema", "description": "Target schema definition", "required": True},
                    {"name": "requirements", "description": "Mapping requirements", "required": False}
                ],
                template="""
Generate a comprehensive schema mapping from {source_schema} to {target_schema}.

Requirements: {requirements}

Please provide:
1. Table-to-table mappings
2. Column-to-column mappings
3. Data type conversions
4. Transformation rules
5. Constraint preservation
6. Performance considerations

Format the output as a structured mapping specification.
"""
            ))
        
        elif intent.intent_type == IntentType.DATA_EXTRACTION:
            prompts.append(MCPPrompt(
                name="data_extraction_assistant",
                description="Assist with data extraction tasks",
                arguments=[
                    {"name": "source", "description": "Data source", "required": True},
                    {"name": "fields", "description": "Fields to extract", "required": False},
                    {"name": "conditions", "description": "Extraction conditions", "required": False}
                ],
                template="""
Generate data extraction logic for:

Source: {source}
Fields: {fields}
Conditions: {conditions}

Please provide:
1. Extraction queries/code
2. Data validation rules
3. Error handling
4. Performance optimization
5. Output format specification

Ensure the extraction maintains data integrity and handles edge cases.
"""
            ))
        
        return prompts
    
    def _generate_schema_resources(self, mcpp_schema: MCPPSchema) -> List[MCPResource]:
        """Generate resources from MCPP schema"""
        
        resources = []
        
        # Schema overview resource
        resources.append(MCPResource(
            uri=f"{self.base_uri}schema/{mcpp_schema.schema_id}",
            name=f"schema_{mcpp_schema.schema_id}",
            description=f"MCPP schema from {mcpp_schema.source_system} to {mcpp_schema.target_system}",
            mimeType="application/json",
            annotations={
                "source_system": mcpp_schema.source_system,
                "target_system": mcpp_schema.target_system,
                "version": mcpp_schema.version,
                "created_at": mcpp_schema.created_at.isoformat()
            }
        ))
        
        # Table resources
        for i, table in enumerate(mcpp_schema.tables):
            resources.append(MCPResource(
                uri=f"{self.base_uri}schema/{mcpp_schema.schema_id}/tables/{i}",
                name=f"table_{table.get('source_table', f'table_{i}')}",
                description=f"Table mapping: {table.get('source_table')} -> {table.get('target_table')}",
                mimeType="application/json",
                annotations=table
            ))
        
        # Relationship resources
        for i, relationship in enumerate(mcpp_schema.relationships):
            resources.append(MCPResource(
                uri=f"{self.base_uri}schema/{mcpp_schema.schema_id}/relationships/{i}",
                name=f"relationship_{i}",
                description=f"Schema relationship {i}",
                mimeType="application/json",
                annotations=relationship
            ))
        
        # Transformation resources
        for i, transformation in enumerate(mcpp_schema.transformations):
            resources.append(MCPResource(
                uri=f"{self.base_uri}schema/{mcpp_schema.schema_id}/transformations/{i}",
                name=f"transformation_{i}",
                description=f"Data transformation {i}",
                mimeType="application/json",
                annotations=transformation
            ))
        
        return resources
    
    def _generate_schema_tools(self, mcpp_schema: MCPPSchema) -> List[MCPTool]:
        """Generate tools from MCPP schema"""
        
        tools = []
        
        # Schema validation tool
        tools.append(MCPTool(
            name="validate_schema",
            description="Validate schema mapping and transformations",
            inputSchema={
                "type": "object",
                "properties": {
                    "schema_id": {"type": "string", "description": "Schema ID to validate"},
                    "validation_rules": {"type": "object", "description": "Custom validation rules"}
                },
                "required": ["schema_id"]
            }
        ))
        
        # Migration generator tool
        tools.append(MCPTool(
            name="generate_migration",
            description="Generate migration scripts from schema",
            inputSchema={
                "type": "object",
                "properties": {
                    "schema_id": {"type": "string", "description": "Schema ID"},
                    "output_format": {"type": "string", "enum": ["sql", "python", "yaml"]},
                    "include_data": {"type": "boolean", "description": "Include data migration"}
                },
                "required": ["schema_id", "output_format"]
            }
        ))
        
        # Data lineage tool
        tools.append(MCPTool(
            name="trace_lineage",
            description="Trace data lineage through transformations",
            inputSchema={
                "type": "object",
                "properties": {
                    "schema_id": {"type": "string", "description": "Schema ID"},
                    "table_name": {"type": "string", "description": "Table to trace"},
                    "column_name": {"type": "string", "description": "Column to trace"}
                },
                "required": ["schema_id"]
            }
        ))
        
        return tools
    
    def _generate_schema_prompts(self, mcpp_schema: MCPPSchema) -> List[MCPPrompt]:
        """Generate prompts from MCPP schema"""
        
        prompts = []
        
        # Migration planning prompt
        prompts.append(MCPPrompt(
            name="migration_planner",
            description="Plan data migration based on schema",
            arguments=[
                {"name": "schema_id", "description": "Schema ID", "required": True},
                {"name": "migration_type", "description": "Type of migration", "required": False},
                {"name": "constraints", "description": "Migration constraints", "required": False}
            ],
            template="""
Plan a data migration for schema {schema_id}:

Source System: {source_system}
Target System: {target_system}
Migration Type: {migration_type}
Constraints: {constraints}

Schema Details:
- Tables: {table_count}
- Relationships: {relationship_count}
- Transformations: {transformation_count}

Please provide:
1. Migration strategy
2. Execution phases
3. Risk assessment
4. Rollback plan
5. Testing approach
6. Performance considerations

Ensure data integrity and minimal downtime.
"""
        ))
        
        return prompts
    
    def export_mcp_json(self, mcp_server: MCPServer) -> str:
        """Export MCP server configuration as JSON"""
        
        mcp_dict = {
            "mcpVersion": "2024-11-05",
            "server": {
                "name": mcp_server.name,
                "version": mcp_server.version
            },
            "capabilities": mcp_server.capabilities,
            "resources": [asdict(resource) for resource in mcp_server.resources],
            "tools": [asdict(tool) for tool in mcp_server.tools],
            "prompts": [asdict(prompt) for prompt in mcp_server.prompts],
            "metadata": mcp_server.metadata
        }
        
        return json.dumps(mcp_dict, indent=2, default=str)
    
    def export_mcp_yaml(self, mcp_server: MCPServer) -> str:
        """Export MCP server configuration as YAML"""
        
        mcp_dict = {
            "mcpVersion": "2024-11-05",
            "server": {
                "name": mcp_server.name,
                "version": mcp_server.version
            },
            "capabilities": mcp_server.capabilities,
            "resources": [asdict(resource) for resource in mcp_server.resources],
            "tools": [asdict(tool) for tool in mcp_server.tools],
            "prompts": [asdict(prompt) for prompt in mcp_server.prompts],
            "metadata": mcp_server.metadata
        }
        
        return yaml.dump(mcp_dict, default_flow_style=False, default=str)

# Integration with existing Sakthi components
class SakthiMCPIntegration:
    """Integration layer for Sakthi platform with MCP"""
    
    def __init__(self, sakthi_engine, genai_agent):
        self.sakthi_engine = sakthi_engine
        self.genai_agent = genai_agent
        self.mcp_generator = SakthiMCPGenerator()
    
    def process_nlp_to_mcp(self, input_text: str, 
                          output_format: str = "json",
                          context: Optional[Dict[str, Any]] = None) -> str:
        """Process NLP input and generate MCP format"""
        
        logger.info("Processing NLP input to MCP format")
        
        # Process with Sakthi
        sakthi_output = self.sakthi_engine.process(input_text, context=context)
        
        # Extract intent from Sakthi output
        intent = self.sakthi_engine.parser.parse(input_text, context)
        
        # Generate MCP configuration
        mcp_server = self.mcp_generator.generate_mcp_from_intent(intent)
        
        # Export in requested format
        if output_format.lower() == "yaml":
            return self.mcp_generator.export_mcp_yaml(mcp_server)
        else:
            return self.mcp_generator.export_mcp_json(mcp_server)
    
    def process_schema_to_mcp(self, mcpp_schema: MCPPSchema, 
                            output_format: str = "json") -> str:
        """Process MCPP schema and generate MCP format"""
        
        logger.info(f"Processing schema {mcpp_schema.schema_id} to MCP format")
        
        # Generate MCP configuration from schema
        mcp_server = self.mcp_generator.generate_mcp_from_schema(mcpp_schema)
        
        # Export in requested format
        if output_format.lower() == "yaml":
            return self.mcp_generator.export_mcp_yaml(mcp_server)
        else:
            return self.mcp_generator.export_mcp_json(mcp_server)
    
    async def process_modeling_request_to_mcp(self, request: Dict[str, Any],
                                            output_format: str = "json") -> str:
        """Process complete modeling request and generate MCP"""
        
        logger.info("Processing modeling request to MCP format")
        
        # Process with GenAI agent
        result = await self.genai_agent.process_modeling_request(request)
        
        # Extract MCPP schema if available
        if result.get('mcpp_schema'):
            mcpp_schema = MCPPSchema(**result['mcpp_schema'])
            return self.process_schema_to_mcp(mcpp_schema, output_format)
        else:
            # Fallback to basic MCP generation
            basic_intent = SakthiIntent(
                intent_type=IntentType.ANALYSIS,
                confidence=0.8,
                entities={},
                source_format="unknown",
                target_format="json",
                parameters=request
            )
            
            mcp_server = self.mcp_generator.generate_mcp_from_intent(basic_intent)
            
            if output_format.lower() == "yaml":
                return self.mcp_generator.export_mcp_yaml(mcp_server)
            else:
                return self.mcp_generator.export_mcp_json(mcp_server)

# Example usage and testing
if __name__ == "__main__":
    from core import SakthiEngine
    
    # Initialize components
    sakthi_engine = SakthiEngine()
    mcp_generator = SakthiMCPGenerator()
    
    # Test NLP to MCP conversion
    test_inputs = [
        "Convert Oracle HR schema to BigQuery, maintain referential integrity",
        "Extract quarterly revenue data from financial_reports.pdf",
        "Map customer table from MySQL to PostgreSQL"
    ]
    
    print("=== Sakthi NLP to MCP Generation Demo ===\n")
    
    for i, input_text in enumerate(test_inputs, 1):
        print(f"Test {i}: {input_text}")
        print("-" * 60)
        
        # Process with Sakthi
        intent = sakthi_engine.parser.parse(input_text)
        
        # Generate MCP
        mcp_server = mcp_generator.generate_mcp_from_intent(intent)
        
        # Export as JSON
        mcp_json = mcp_generator.export_mcp_json(mcp_server)
        
        print("Generated MCP Configuration:")
        print(mcp_json[:800] + "..." if len(mcp_json) > 800 else mcp_json)
        print("\n" + "="*70 + "\n")
    
    print("MCP generation complete!")