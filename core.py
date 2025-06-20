# sakthi_language/core.py
"""
Sakthi Language Core Implementation
Natural Language to Structured Output Processor
"""

import re
import json
import yaml
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OutputFormat(Enum):
    JSON = "json"
    YAML = "yaml"
    SQL = "sql"
    PYTHON = "python"
    API_CALL = "api_call"
    CONFIG = "config"

class IntentType(Enum):
    DATA_EXTRACTION = "data_extraction"
    SCHEMA_MAPPING = "schema_mapping"
    TRANSFORMATION = "transformation"
    MIGRATION = "migration"
    ANALYSIS = "analysis"
    CONFIGURATION = "configuration"

@dataclass
class SakthiIntent:
    intent_type: IntentType
    confidence: float
    entities: Dict[str, Any]
    source_format: str
    target_format: str
    parameters: Dict[str, Any]

@dataclass
class SakthiOutput:
    format: OutputFormat
    content: str
    metadata: Dict[str, Any]
    confidence: float
    validation_status: bool

class SakthiParser:
    """Core Sakthi Language Parser"""
    
    def __init__(self):
        self.intent_patterns = self._load_intent_patterns()
        self.entity_extractors = self._load_entity_extractors()
        self.format_generators = self._load_format_generators()
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load predefined intent recognition patterns"""
        return {
            "data_extraction": [
                r"extract\s+(.+?)\s+from\s+(.+)",
                r"get\s+(.+?)\s+data\s+from\s+(.+)",
                r"pull\s+(.+?)\s+from\s+(.+)",
                r"retrieve\s+(.+?)\s+from\s+(.+)"
            ],
            "schema_mapping": [
                r"map\s+(.+?)\s+to\s+(.+)",
                r"convert\s+(.+?)\s+schema\s+to\s+(.+)",
                r"transform\s+(.+?)\s+structure\s+to\s+(.+)",
                r"migrate\s+(.+?)\s+to\s+(.+)"
            ],
            "transformation": [
                r"transform\s+(.+?)\s+into\s+(.+)",
                r"change\s+(.+?)\s+to\s+(.+)",
                r"modify\s+(.+?)\s+format\s+to\s+(.+)",
                r"restructure\s+(.+?)\s+as\s+(.+)"
            ],
            "analysis": [
                r"analyze\s+(.+)",
                r"examine\s+(.+)",
                r"study\s+(.+)",
                r"investigate\s+(.+)"
            ]
        }
    
    def _load_entity_extractors(self) -> Dict[str, str]:
        """Load entity extraction patterns"""
        return {
            "database": r"\b(oracle|mysql|postgresql|sql\s*server|mongodb|cassandra|bigquery|snowflake|redshift)\b",
            "file_format": r"\b(pdf|docx|csv|xlsx|json|xml|yaml|txt)\b",
            "column_name": r"['\"`]([^'`\"]+)['\"`]",
            "table_name": r"\b([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*|\b[a-zA-Z_][a-zA-Z0-9_]*)\b",
            "data_type": r"\b(varchar|int|integer|decimal|float|date|timestamp|boolean|text|blob)\b",
            "field_name": r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b"
        }
    
    def _load_format_generators(self) -> Dict[str, Any]:
        """Load output format generators"""
        return {
            OutputFormat.JSON: self._generate_json,
            OutputFormat.YAML: self._generate_yaml,
            OutputFormat.SQL: self._generate_sql,
            OutputFormat.PYTHON: self._generate_python,
            OutputFormat.API_CALL: self._generate_api_call,
            OutputFormat.CONFIG: self._generate_config
        }
    
    def parse(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> SakthiIntent:
        """Parse natural language input into structured intent"""
        logger.info(f"Parsing input: {input_text[:100]}...")
        
        # Clean and normalize input
        cleaned_input = self._clean_input(input_text)
        
        # Extract intent
        intent_type, confidence = self._extract_intent(cleaned_input)
        
        # Extract entities
        entities = self._extract_entities(cleaned_input)
        
        # Determine formats
        source_format = self._determine_source_format(cleaned_input, entities)
        target_format = self._determine_target_format(cleaned_input, entities)
        
        # Extract parameters
        parameters = self._extract_parameters(cleaned_input, context or {})
        
        return SakthiIntent(
            intent_type=intent_type,
            confidence=confidence,
            entities=entities,
            source_format=source_format,
            target_format=target_format,
            parameters=parameters
        )
    
    def _clean_input(self, text: str) -> str:
        """Clean and normalize input text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Convert to lowercase for pattern matching
        return text.lower()
    
    def _extract_intent(self, text: str) -> tuple[IntentType, float]:
        """Extract intent from text with confidence score"""
        max_confidence = 0.0
        detected_intent = IntentType.ANALYSIS  # Default
        
        for intent_name, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    confidence = len(match.group(0)) / len(text)
                    if confidence > max_confidence:
                        max_confidence = confidence
                        detected_intent = IntentType(intent_name)
        
        return detected_intent, min(max_confidence * 2, 1.0)  # Normalize confidence
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from text"""
        entities = {}
        
        for entity_type, pattern in self.entity_extractors.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches
        
        return entities
    
    def _determine_source_format(self, text: str, entities: Dict[str, Any]) -> str:
        """Determine source data format"""
        if 'file_format' in entities:
            return entities['file_format'][0]
        elif 'database' in entities:
            return entities['database'][0]
        else:
            return 'unknown'
    
    def _determine_target_format(self, text: str, entities: Dict[str, Any]) -> str:
        """Determine target output format"""
        # Look for explicit format mentions
        format_keywords = {
            'json': 'json',
            'yaml': 'yaml', 
            'sql': 'sql',
            'python': 'python',
            'api': 'api_call',
            'config': 'config'
        }
        
        for keyword, format_type in format_keywords.items():
            if keyword in text:
                return format_type
        
        return 'json'  # Default
    
    def _extract_parameters(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract operation parameters"""
        parameters = {}
        
        # Extract common parameters
        if 'maintain' in text and 'integrity' in text:
            parameters['maintain_referential_integrity'] = True
        
        if 'include' in text and 'constraint' in text:
            parameters['include_constraints'] = True
        
        if 'batch' in text or 'bulk' in text:
            parameters['batch_processing'] = True
        
        # Add context parameters
        parameters.update(context)
        
        return parameters
    
    def generate_output(self, intent: SakthiIntent, output_format: OutputFormat) -> SakthiOutput:
        """Generate structured output based on intent"""
        logger.info(f"Generating {output_format.value} output for {intent.intent_type.value}")
        
        generator = self.format_generators.get(output_format)
        if not generator:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        content = generator(intent)
        
        return SakthiOutput(
            format=output_format,
            content=content,
            metadata={
                'intent_type': intent.intent_type.value,
                'confidence': intent.confidence,
                'generated_at': datetime.now().isoformat(),
                'source_format': intent.source_format,
                'target_format': intent.target_format
            },
            confidence=intent.confidence,
            validation_status=True  # TODO: Implement validation
        )
    
    def _generate_json(self, intent: SakthiIntent) -> str:
        """Generate JSON output"""
        if intent.intent_type == IntentType.SCHEMA_MAPPING:
            schema = {
                "mapping": {
                    "source": intent.source_format,
                    "target": intent.target_format,
                    "tables": [],
                    "transformations": []
                }
            }
            return json.dumps(schema, indent=2)
        
        return json.dumps({"intent": intent.intent_type.value, "entities": intent.entities}, indent=2)
    
    def _generate_yaml(self, intent: SakthiIntent) -> str:
        """Generate YAML output"""
        data = {
            "intent": intent.intent_type.value,
            "entities": intent.entities,
            "parameters": intent.parameters
        }
        return yaml.dump(data, default_flow_style=False)
    
    def _generate_sql(self, intent: SakthiIntent) -> str:
        """Generate SQL output"""
        if intent.intent_type == IntentType.SCHEMA_MAPPING:
            # Generate schema migration SQL
            return self._generate_migration_sql(intent)
        elif intent.intent_type == IntentType.DATA_EXTRACTION:
            return self._generate_extraction_sql(intent)
        
        return "-- SQL generation not implemented for this intent type"
    
    def _generate_migration_sql(self, intent: SakthiIntent) -> str:
        """Generate SQL for schema migration"""
        sql_parts = []
        
        # Add header comment
        sql_parts.append(f"-- Schema migration from {intent.source_format} to {intent.target_format}")
        sql_parts.append(f"-- Generated at {datetime.now().isoformat()}")
        sql_parts.append("")
        
        # Add table creation templates
        if 'table_name' in intent.entities:
            for table in intent.entities['table_name']:
                sql_parts.append(f"CREATE TABLE {table} (")
                sql_parts.append("    -- Add column definitions here")
                sql_parts.append(");")
                sql_parts.append("")
        
        return "\n".join(sql_parts)
    
    def _generate_extraction_sql(self, intent: SakthiIntent) -> str:
        """Generate SQL for data extraction"""
        if 'table_name' in intent.entities:
            table = intent.entities['table_name'][0]
            return f"SELECT * FROM {table};"
        
        return "SELECT * FROM table_name;"
    
    def _generate_python(self, intent: SakthiIntent) -> str:
        """Generate Python code"""
        code_parts = []
        
        code_parts.append("# Generated Python code for Sakthi intent")
        code_parts.append(f"# Intent: {intent.intent_type.value}")
        code_parts.append("")
        code_parts.append("import pandas as pd")
        code_parts.append("import json")
        code_parts.append("")
        
        if intent.intent_type == IntentType.DATA_EXTRACTION:
            code_parts.append("def extract_data():")
            code_parts.append("    # Implementation for data extraction")
            code_parts.append("    pass")
        
        return "\n".join(code_parts)
    
    def _generate_api_call(self, intent: SakthiIntent) -> str:
        """Generate API call configuration"""
        api_config = {
            "method": "GET",
            "url": "/api/data",
            "headers": {
                "Content-Type": "application/json"
            },
            "params": intent.parameters
        }
        return json.dumps(api_config, indent=2)
    
    def _generate_config(self, intent: SakthiIntent) -> str:
        """Generate configuration file"""
        config = {
            "sakthi_config": {
                "intent": intent.intent_type.value,
                "source": intent.source_format,
                "target": intent.target_format,
                "parameters": intent.parameters
            }
        }
        return yaml.dump(config, default_flow_style=False)

class SakthiEngine:
    """Main Sakthi Language Engine"""
    
    def __init__(self):
        self.parser = SakthiParser()
        self.processors = {}
        self.validators = {}
    
    def process(self, input_text: str, output_format: OutputFormat = OutputFormat.JSON, 
                context: Optional[Dict[str, Any]] = None) -> SakthiOutput:
        """Process natural language input and generate structured output"""
        
        # Parse input to extract intent
        intent = self.parser.parse(input_text, context)
        
        # Generate output
        output = self.parser.generate_output(intent, output_format)
        
        # Log processing
        logger.info(f"Processed intent: {intent.intent_type.value} with confidence: {intent.confidence:.2f}")
        
        return output
    
    def batch_process(self, inputs: List[str], output_format: OutputFormat = OutputFormat.JSON) -> List[SakthiOutput]:
        """Process multiple inputs in batch"""
        results = []
        
        for input_text in inputs:
            try:
                result = self.process(input_text, output_format)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing input '{input_text}': {str(e)}")
                # Add error result
                results.append(SakthiOutput(
                    format=output_format,
                    content=f"Error: {str(e)}",
                    metadata={"error": True},
                    confidence=0.0,
                    validation_status=False
                ))
        
        return results

# Example usage and testing
if __name__ == "__main__":
    # Initialize Sakthi engine
    sakthi = SakthiEngine()
    
    # Test cases
    test_inputs = [
        "Convert Oracle HR schema to BigQuery, maintain referential integrity",
        "Extract quarterly revenue data from financial_reports.pdf",
        "Map customer table from MySQL to PostgreSQL",
        "Transform JSON data into CSV format",
        "Analyze sales data from last quarter"
    ]
    
    print("=== Sakthi Language Processing Demo ===\n")
    
    for i, input_text in enumerate(test_inputs, 1):
        print(f"Test {i}: {input_text}")
        print("-" * 50)
        
        # Process with different output formats
        for output_format in [OutputFormat.JSON, OutputFormat.SQL, OutputFormat.PYTHON]:
            try:
                result = sakthi.process(input_text, output_format)
                print(f"\n{output_format.value.upper()} Output:")
                print(result.content)
                print(f"Confidence: {result.confidence:.2f}")
            except Exception as e:
                print(f"Error with {output_format.value}: {str(e)}")
        
        print("\n" + "="*70 + "\n")