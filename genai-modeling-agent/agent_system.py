# genai_modeling_agent/agent_system.py
"""
GenAI-powered modeling agent using:
- MCPP based schema/mapping generation
- LangGraph for audit-aware orchestration
- AutoGen for Extractor, Mapper, Verifier agents
- Metadata APIs (Atlan, Collibra) for real-time tagging
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import uuid

# AutoGen imports
try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
except ImportError:
    logging.warning("AutoGen not available, using mock implementation")

# LangGraph imports  
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.prebuilt import ToolExecutor, ToolInvocation
except ImportError:
    logging.warning("LangGraph not available, using alternative implementation")

# OpenAI/LangChain imports
try:
    import openai
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
except ImportError:
    logging.warning("LangChain/OpenAI not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentRole(Enum):
    EXTRACTOR = "extractor"
    MAPPER = "mapper" 
    VERIFIER = "verifier"
    ORCHESTRATOR = "orchestrator"

class ProcessingState(Enum):
    INITIALIZED = "initialized"
    EXTRACTING = "extracting"
    MAPPING = "mapping"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class MCPPSchema:
    """Model-Centric Processing Pipeline Schema"""
    schema_id: str
    source_system: str
    target_system: str
    tables: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    transformations: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    version: str = "1.0"
    created_at: datetime = datetime.now()

@dataclass 
class ProcessingContext:
    """Context for processing workflow"""
    session_id: str
    input_data: Dict[str, Any]
    current_state: ProcessingState
    agents_involved: List[str]
    audit_trail: List[Dict[str, Any]]
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime = datetime.now()

class ExtractorAgent:
    """Data source analysis and extraction agent"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.agent_id = f"extractor_{uuid.uuid4().hex[:8]}"
        
        # Initialize AutoGen agent
        self.agent = AssistantAgent(
            name="DataExtractor",
            system_message="""You are a data extraction specialist. Your role is to:
            1. Analyze input data sources (databases, files, APIs)
            2. Identify data structures, schemas, and relationships
            3. Extract metadata and lineage information
            4. Provide detailed analysis of data quality and completeness
            
            Always provide structured JSON output with your findings.""",
            llm_config=llm_config
        )
    
    async def extract_schema(self, source_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract schema information from source data"""
        
        logger.info(f"Extractor {self.agent_id} starting schema extraction")
        
        try:
            # Prepare extraction prompt
            extraction_prompt = f"""
            Analyze the following data source and extract schema information:
            
            Source Type: {source_data.get('type', 'unknown')}
            Data Sample: {json.dumps(source_data.get('sample', {}), indent=2)[:1000]}
            
            Please provide:
            1. Table/collection names
            2. Column/field definitions with data types
            3. Primary keys and indexes
            4. Foreign key relationships
            5. Constraints and validations
            6. Data quality assessment
            
            Format your response as structured JSON.
            """
            
            # Get response from LLM
            response = await self._get_llm_response(extraction_prompt)
            
            # Parse and structure the response
            extracted_schema = self._parse_extraction_response(response)
            
            # Add metadata
            extracted_schema['extraction_metadata'] = {
                'agent_id': self.agent_id,
                'extraction_time': datetime.now().isoformat(),
                'source_type': source_data.get('type'),
                'confidence_score': self._calculate_confidence(extracted_schema)
            }
            
            logger.info(f"Schema extraction completed with confidence: {extracted_schema['extraction_metadata']['confidence_score']}")
            
            return extracted_schema
            
        except Exception as e:
            logger.error(f"Schema extraction failed: {str(e)}")
            return {
                'error': str(e),
                'agent_id': self.agent_id,
                'extraction_time': datetime.now().isoformat()
            }
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM"""
        # Mock implementation - replace with actual LLM call
        return json.dumps({
            "tables": [
                {
                    "name": "users",
                    "columns": [
                        {"name": "id", "type": "INTEGER", "primary_key": True},
                        {"name": "username", "type": "VARCHAR(50)", "nullable": False},
                        {"name": "email", "type": "VARCHAR(100)", "nullable": False}
                    ]
                }
            ],
            "relationships": [],
            "quality_score": 0.85
        })
    
    def _parse_extraction_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback parsing
            return {
                "tables": [],
                "relationships": [],
                "quality_score": 0.5,
                "raw_response": response
            }
    
    def _calculate_confidence(self, schema: Dict[str, Any]) -> float:
        """Calculate confidence score for extracted schema"""
        confidence = 0.0
        
        # Base confidence
        if schema.get('tables'):
            confidence += 0.4
        
        if schema.get('relationships'):
            confidence += 0.2
        
        if schema.get('quality_score', 0) > 0.7:
            confidence += 0.3
        
        # Additional factors
        table_count = len(schema.get('tables', []))
        if table_count > 0:
            confidence += min(table_count * 0.05, 0.1)
        
        return min(confidence, 1.0)

class MapperAgent:
    """Schema mapping and transformation agent"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.agent_id = f"mapper_{uuid.uuid4().hex[:8]}"
        
        self.agent = AssistantAgent(
            name="SchemaMapper",
            system_message="""You are a schema mapping specialist. Your role is to:
            1. Map source schemas to target schemas
            2. Identify transformation requirements
            3. Generate mapping rules and logic
            4. Handle data type conversions and constraints
            5. Optimize for performance and data integrity
            
            Provide detailed mapping specifications with transformation logic.""",
            llm_config=llm_config
        )
    
    async def generate_mapping(self, source_schema: Dict[str, Any], 
                             target_schema: Dict[str, Any],
                             mapping_requirements: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate schema mapping between source and target"""
        
        logger.info(f"Mapper {self.agent_id} generating schema mapping")
        
        try:
            mapping_prompt = f"""
            Generate a comprehensive mapping between source and target schemas:
            
            SOURCE SCHEMA:
            {json.dumps(source_schema, indent=2)[:1500]}
            
            TARGET SCHEMA:
            {json.dumps(target_schema, indent=2)[:1500]}
            
            REQUIREMENTS:
            {json.dumps(mapping_requirements or {}, indent=2)}
            
            Please provide:
            1. Table-to-table mappings
            2. Column-to-column mappings with transformations
            3. Data type conversion rules
            4. Default values and null handling
            5. Business rule transformations
            6. Performance optimization suggestions
            
            Format as structured JSON with detailed transformation logic.
            """
            
            response = await self._get_llm_response(mapping_prompt)
            mapping_result = self._parse_mapping_response(response)
            
            # Generate MCPP schema
            mcpp_schema = self._generate_mcpp_schema(source_schema, target_schema, mapping_result)
            
            # Add mapping metadata
            mapping_result['mapping_metadata'] = {
                'agent_id': self.agent_id,
                'mapping_time': datetime.now().isoformat(),
                'complexity_score': self._calculate_mapping_complexity(mapping_result),
                'mcpp_schema_id': mcpp_schema.schema_id
            }
            
            mapping_result['mcpp_schema'] = asdict(mcpp_schema)
            
            logger.info(f"Schema mapping completed with complexity score: {mapping_result['mapping_metadata']['complexity_score']}")
            
            return mapping_result
            
        except Exception as e:
            logger.error(f"Schema mapping failed: {str(e)}")
            return {
                'error': str(e),
                'agent_id': self.agent_id,
                'mapping_time': datetime.now().isoformat()
            }
    
    def _generate_mcpp_schema(self, source_schema: Dict[str, Any], 
                            target_schema: Dict[str, Any], 
                            mapping_result: Dict[str, Any]) -> MCPPSchema:
        """Generate MCPP-based schema representation"""
        
        schema_id = f"mcpp_{uuid.uuid4().hex[:12]}"
        
        # Extract source and target system info
        source_system = source_schema.get('system_type', 'unknown')
        target_system = target_schema.get('system_type', 'unknown')
        
        # Process table mappings
        tables = []
        for table_mapping in mapping_result.get('table_mappings', []):
            table_info = {
                'source_table': table_mapping.get('source_table'),
                'target_table': table_mapping.get('target_table'),
                'columns': table_mapping.get('column_mappings', []),
                'constraints': table_mapping.get('constraints', [])
            }
            tables.append(table_info)
        
        # Process relationships
        relationships = mapping_result.get('relationships', [])
        
        # Process transformations
        transformations = mapping_result.get('transformations', [])
        
        # Metadata
        metadata = {
            'mapping_complexity': mapping_result.get('mapping_metadata', {}).get('complexity_score', 0.5),
            'source_tables': len(source_schema.get('tables', [])),
            'target_tables': len(target_schema.get('tables', [])),
            'transformation_count': len(transformations)
        }
        
        return MCPPSchema(
            schema_id=schema_id,
            source_system=source_system,
            target_system=target_system,
            tables=tables,
            relationships=relationships,
            transformations=transformations,
            metadata=metadata
        )
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM"""
        # Mock implementation
        return json.dumps({
            "table_mappings": [
                {
                    "source_table": "users",
                    "target_table": "customer",
                    "column_mappings": [
                        {"source": "id", "target": "customer_id", "transformation": "direct"},
                        {"source": "username", "target": "customer_name", "transformation": "direct"},
                        {"source": "email", "target": "email_address", "transformation": "direct"}
                    ]
                }
            ],
            "transformations": [
                {"type": "rename", "description": "Rename users table to customer"}
            ],
            "complexity_score": 0.3
        })
    
    def _parse_mapping_response(self, response: str) -> Dict[str, Any]:
        """Parse mapping response"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "table_mappings": [],
                "transformations": [],
                "relationships": [],
                "raw_response": response
            }
    
    def _calculate_mapping_complexity(self, mapping: Dict[str, Any]) -> float:
        """Calculate mapping complexity score"""
        complexity = 0.0
        
        # Base complexity from table count
        table_count = len(mapping.get('table_mappings', []))
        complexity += table_count * 0.1
        
        # Transformation complexity
        transformation_count = len(mapping.get('transformations', []))
        complexity += transformation_count * 0.15
        
        # Relationship complexity
        relationship_count = len(mapping.get('relationships', []))
        complexity += relationship_count * 0.1
        
        return min(complexity, 1.0)

class VerifierAgent:
    """Quality assurance and validation agent"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.agent_id = f"verifier_{uuid.uuid4().hex[:8]}"
        
        self.agent = AssistantAgent(
            name="QualityVerifier",
            system_message="""You are a quality assurance specialist. Your role is to:
            1. Validate schema mappings and transformations
            2. Check for data integrity issues
            3. Verify business rule compliance
            4. Identify potential performance issues
            5. Ensure completeness and accuracy
            
            Provide comprehensive validation reports with recommendations.""",
            llm_config=llm_config
        )
    
    async def verify_mapping(self, mapping_result: Dict[str, Any], 
                           validation_rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Verify and validate mapping results"""
        
        logger.info(f"Verifier {self.agent_id} starting mapping verification")
        
        try:
            verification_prompt = f"""
            Verify the following schema mapping for accuracy and completeness:
            
            MAPPING RESULT:
            {json.dumps(mapping_result, indent=2)[:2000]}
            
            VALIDATION RULES:
            {json.dumps(validation_rules or {}, indent=2)}
            
            Please check:
            1. Mapping completeness (all source tables/columns mapped)
            2. Data type compatibility
            3. Constraint preservation
            4. Business rule compliance
            5. Performance implications
            6. Data integrity risks
            
            Provide validation score (0-1) and detailed findings.
            """
            
            response = await self._get_llm_response(verification_prompt)
            verification_result = self._parse_verification_response(response)
            
            # Add verification metadata
            verification_result['verification_metadata'] = {
                'agent_id': self.agent_id,
                'verification_time': datetime.now().isoformat(),
                'validation_score': verification_result.get('validation_score', 0.5),
                'issues_found': len(verification_result.get('issues', []))
            }
            
            logger.info(f"Mapping verification completed with score: {verification_result['verification_metadata']['validation_score']}")
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Mapping verification failed: {str(e)}")
            return {
                'error': str(e),
                'agent_id': self.agent_id,
                'verification_time': datetime.now().isoformat(),
                'validation_score': 0.0
            }
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM"""
        # Mock implementation
        return json.dumps({
            "validation_score": 0.85,
            "issues": [
                {"severity": "low", "description": "Consider adding index on customer_id"},
                {"severity": "medium", "description": "Email validation rule missing"}
            ],
            "recommendations": [
                "Add data validation for email format",
                "Consider partitioning strategy for large tables"
            ],
            "completeness_check": {
                "mapped_tables": "100%",
                "mapped_columns": "95%",
                "missing_mappings": ["users.created_at"]
            }
        })
    
    def _parse_verification_response(self, response: str) -> Dict[str, Any]:
        """Parse verification response"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "validation_score": 0.5,
                "issues": [],
                "recommendations": [],
                "raw_response": response
            }

class LangGraphOrchestrator:
    """LangGraph-based workflow orchestration with audit trail"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.checkpointer = None  # SqliteSaver() if available
        self.workflow_graph = self._build_workflow_graph()
    
    def _build_workflow_graph(self):
        """Build the processing workflow graph"""
        
        # Define workflow state
        class WorkflowState:
            session_id: str
            input_data: Dict[str, Any]
            extracted_schema: Dict[str, Any]
            mapping_result: Dict[str, Any]
            verification_result: Dict[str, Any]
            final_result: Dict[str, Any]
            audit_trail: List[Dict[str, Any]]
            current_step: str
        
        # Create workflow graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("extract", self._extract_node)
        workflow.add_node("map", self._map_node)
        workflow.add_node("verify", self._verify_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Add edges
        workflow.add_edge("extract", "map")
        workflow.add_edge("map", "verify") 
        workflow.add_edge("verify", "finalize")
        workflow.add_edge("finalize", END)
        
        # Set entry point
        workflow.set_entry_point("extract")
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def _extract_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extraction node processing"""
        
        # Add audit entry
        audit_entry = {
            'step': 'extract',
            'timestamp': datetime.now().isoformat(),
            'agent': 'extractor'
        }
        
        # Initialize extractor agent
        extractor = ExtractorAgent(self.llm_config)
        
        # Perform extraction
        extracted_schema = await extractor.extract_schema(state['input_data'])
        
        # Update audit
        audit_entry['result'] = 'success' if 'error' not in extracted_schema else 'failed'
        audit_entry['details'] = extracted_schema.get('extraction_metadata', {})
        
        # Update state
        state['extracted_schema'] = extracted_schema
        state['audit_trail'].append(audit_entry)
        state['current_step'] = 'extract_completed'
        
        return state
    
    async def _map_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Mapping node processing"""
        
        audit_entry = {
            'step': 'map',
            'timestamp': datetime.now().isoformat(),
            'agent': 'mapper'
        }
        
        # Initialize mapper agent
        mapper = MapperAgent(self.llm_config)
        
        # Get target schema from input or use default
        target_schema = state['input_data'].get('target_schema', {})
        
        # Perform mapping
        mapping_result = await mapper.generate_mapping(
            state['extracted_schema'], 
            target_schema,
            state['input_data'].get('mapping_requirements')
        )
        
        # Update audit
        audit_entry['result'] = 'success' if 'error' not in mapping_result else 'failed'
        audit_entry['details'] = mapping_result.get('mapping_metadata', {})
        
        # Update state
        state['mapping_result'] = mapping_result
        state['audit_trail'].append(audit_entry)
        state['current_step'] = 'map_completed'
        
        return state
    
    async def _verify_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Verification node processing"""
        
        audit_entry = {
            'step': 'verify',
            'timestamp': datetime.now().isoformat(),
            'agent': 'verifier'
        }
        
        # Initialize verifier agent
        verifier = VerifierAgent(self.llm_config)
        
        # Perform verification
        verification_result = await verifier.verify_mapping(
            state['mapping_result'],
            state['input_data'].get('validation_rules')
        )
        
        # Update audit
        audit_entry['result'] = 'success' if 'error' not in verification_result else 'failed'
        audit_entry['details'] = verification_result.get('verification_metadata', {})
        
        # Update state
        state['verification_result'] = verification_result
        state['audit_trail'].append(audit_entry)
        state['current_step'] = 'verify_completed'
        
        return state
    
    async def _finalize_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Finalization node processing"""
        
        audit_entry = {
            'step': 'finalize',
            'timestamp': datetime.now().isoformat(),
            'agent': 'orchestrator'
        }
        
        # Compile final result
        final_result = {
            'session_id': state['session_id'],
            'processing_summary': {
                'extraction': state['extracted_schema'],
                'mapping': state['mapping_result'],
                'verification': state['verification_result']
            },
            'mcpp_schema': state['mapping_result'].get('mcpp_schema'),
            'audit_trail': state['audit_trail'],
            'completion_time': datetime.now().isoformat(),
            'overall_success': all(
                'error' not in result for result in [
                    state['extracted_schema'],
                    state['mapping_result'], 
                    state['verification_result']
                ]
            )
        }
        
        # Calculate overall quality score
        extraction_conf = state['extracted_schema'].get('extraction_metadata', {}).get('confidence_score', 0)
        mapping_complex = state['mapping_result'].get('mapping_metadata', {}).get('complexity_score', 0)
        verification_score = state['verification_result'].get('verification_metadata', {}).get('validation_score', 0)
        
        final_result['quality_metrics'] = {
            'extraction_confidence': extraction_conf,
            'mapping_complexity': mapping_complex,
            'verification_score': verification_score,
            'overall_score': (extraction_conf + verification_score - mapping_complex) / 2
        }
        
        # Update audit
        audit_entry['result'] = 'success'
        audit_entry['details'] = final_result['quality_metrics']
        
        # Update state
        state['final_result'] = final_result
        state['audit_trail'].append(audit_entry)
        state['current_step'] = 'completed'
        
        return state
    
    async def execute_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete modeling workflow"""
        
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        
        logger.info(f"Starting workflow execution for session: {session_id}")
        
        # Initialize state
        initial_state = {
            'session_id': session_id,
            'input_data': input_data,
            'extracted_schema': {},
            'mapping_result': {},
            'verification_result': {},
            'final_result': {},
            'audit_trail': [],
            'current_step': 'initialized'
        }
        
        try:
            # Execute workflow
            result = await self.workflow_graph.ainvoke(initial_state)
            
            logger.info(f"Workflow completed successfully for session: {session_id}")
            return result['final_result']
            
        except Exception as e:
            logger.error(f"Workflow execution failed for session {session_id}: {str(e)}")
            return {
                'session_id': session_id,
                'error': str(e),
                'completion_time': datetime.now().isoformat(),
                'overall_success': False
            }

class MetadataIntegration:
    """Integration with metadata management systems (Atlan, Collibra)"""
    
    def __init__(self, atlan_config: Optional[Dict] = None, collibra_config: Optional[Dict] = None):
        self.atlan_config = atlan_config
        self.collibra_config = collibra_config
    
    async def tag_assets(self, mcpp_schema: MCPPSchema, tags: List[str]) -> Dict[str, Any]:
        """Tag assets in metadata management systems"""
        
        results = {}
        
        if self.atlan_config:
            results['atlan'] = await self._tag_atlan_assets(mcpp_schema, tags)
        
        if self.collibra_config:
            results['collibra'] = await self._tag_collibra_assets(mcpp_schema, tags)
        
        return results
    
    async def _tag_atlan_assets(self, schema: MCPPSchema, tags: List[str]) -> Dict[str, Any]:
        """Tag assets in Atlan"""
        # Mock implementation - replace with actual Atlan API calls
        return {
            'status': 'success',
            'tagged_assets': len(schema.tables),
            'tags_applied': tags
        }
    
    async def _tag_collibra_assets(self, schema: MCPPSchema, tags: List[str]) -> Dict[str, Any]:
        """Tag assets in Collibra"""
        # Mock implementation - replace with actual Collibra API calls
        return {
            'status': 'success',
            'tagged_assets': len(schema.tables),
            'tags_applied': tags
        }

class GenAIModelingAgent:
    """Main GenAI modeling agent orchestrating the entire process"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config.get('llm_config', {})
        
        # Initialize components
        self.orchestrator = LangGraphOrchestrator(self.llm_config)
        self.metadata_integration = MetadataIntegration(
            config.get('atlan_config'),
            config.get('collibra_config')
        )
    
    async def process_modeling_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a complete modeling request"""
        
        logger.info("Processing modeling request")
        
        try:
            # Execute workflow
            result = await self.orchestrator.execute_workflow(request)
            
            # Apply metadata tagging if successful
            if result.get('overall_success') and result.get('mcpp_schema'):
                mcpp_schema = MCPPSchema(**result['mcpp_schema'])
                tags = request.get('metadata_tags', ['auto-generated', 'sakthi-processed'])
                
                tagging_result = await self.metadata_integration.tag_assets(mcpp_schema, tags)
                result['metadata_tagging'] = tagging_result
            
            return result
            
        except Exception as e:
            logger.error(f"Modeling request processing failed: {str(e)}")
            return {
                'error': str(e),
                'completion_time': datetime.now().isoformat(),
                'overall_success': False
            }

# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = {
        'llm_config': {
            'model': 'gpt-4',
            'temperature': 0.1,
            'max_tokens': 2000
        },
        'atlan_config': {
            'api_key': 'your-atlan-key',
            'base_url': 'https://your-instance.atlan.com'
        },
        'collibra_config': {
            'api_key': 'your-collibra-key',
            'base_url': 'https://your-instance.collibra.com'
        }
    }
    
    # Initialize agent
    modeling_agent = GenAIModelingAgent(config)
    
    # Sample request
    sample_request = {
        'source_data': {
            'type': 'postgresql',
            'sample': {
                'tables': [
                    {
                        'name': 'users',
                        'columns': [
                            {'name': 'id', 'type': 'INTEGER'},
                            {'name': 'username', 'type': 'VARCHAR(50)'},
                            {'name': 'email', 'type': 'VARCHAR(100)'}
                        ]
                    }
                ]
            }
        },
        'target_schema': {
            'system_type': 'bigquery',
            'requirements': ['maintain_constraints', 'optimize_performance']
        },
        'mapping_requirements': {
            'preserve_relationships': True,
            'data_validation': True
        },
        'metadata_tags': ['migration', 'postgresql-to-bigquery', 'production']
    }
    
    async def run_example():
        print("=== GenAI Modeling Agent Demo ===\n")
        
        result = await modeling_agent.process_modeling_request(sample_request)
        
        print("Processing Result:")
        print(json.dumps(result, indent=2, default=str))
        
        print(f"\nOverall Success: {result.get('overall_success', False)}")
        print(f"Quality Score: {result.get('quality_metrics', {}).get('overall_score', 'N/A')}")
    
    # Run the example
    asyncio.run(run_example())