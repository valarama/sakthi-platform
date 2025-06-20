# mcp_engine.py
"""
Dynamic MCP Generation Engine with LLM Integration
Uses Jinja2 templates and LLM analysis for dynamic MCP creation
"""

import json
import hashlib
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import requests
import os
from dataclasses import dataclass
from jinja2 import Environment, FileSystemLoader, Template
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMAnalysisResult:
    """Result from LLM analysis"""
    description: str
    purpose: str
    features: List[str]
    complexity: str
    confidence: float
    processing_time: float
    conversion_strategy: Dict[str, Any]

@dataclass
class FileMetadata:
    """Metadata about the processed file"""
    filename: str
    file_size: int
    file_hash: str
    upload_timestamp: datetime
    processing_time_ms: float

class DynamicMCPEngine:
    """Dynamic MCP Generation Engine with LLM-powered analysis"""
    
    def __init__(self, config_path: str = "mcp_templates/oracle_bigquery_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.jinja_env = Environment(
            loader=FileSystemLoader('mcp_templates'),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.llm_endpoints = self._get_llm_endpoints()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load MCP template configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails"""
        return {
            "mcp_template_config": {
                "version": "1.0.0",
                "template_type": "oracle_to_bigquery",
                "dynamic_fields": {"requires_llm_analysis": True}
            }
        }
    
    def _get_llm_endpoints(self) -> List[str]:
        """Get LLM endpoints from environment"""
        endpoints_str = os.getenv('LLM_ENDPOINTS', 'http://10.100.15.67:1138/v1/chat/completions')
        return [ep.strip() for ep in endpoints_str.split(',') if ep.strip()]
    
    async def analyze_oracle_file_with_llm(self, oracle_sql: str, filename: str) -> LLMAnalysisResult:
        """Analyze Oracle file using LLM with retry mechanism"""
        
        start_time = time.time()
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Get analysis prompts from config
                prompts_config = self.config["mcp_template_config"]["llm_analysis_prompts"]
                
                # Analyze file description
                description_result = await self._call_llm_analysis(
                    oracle_sql, 
                    prompts_config["file_description"]
                )
                
                # Detect features
                features_result = await self._call_llm_analysis(
                    oracle_sql,
                    prompts_config["feature_detection"]
                )
                
                # Get conversion strategy
                strategy_result = await self._call_llm_analysis(
                    oracle_sql,
                    prompts_config["conversion_strategy"]
                )
                
                # Parse LLM responses
                description_data = self._parse_llm_json(description_result)
                features_data = self._parse_llm_json(features_result)
                strategy_data = self._parse_llm_json(strategy_result)
                
                # Calculate confidence based on response quality
                confidence = self._calculate_analysis_confidence(
                    description_data, features_data, strategy_data
                )
                
                processing_time = time.time() - start_time
                
                return LLMAnalysisResult(
                    description=description_data.get('description', f'Oracle SQL file: {filename}'),
                    purpose=description_data.get('purpose', 'Database operations'),
                    features=features_data if isinstance(features_data, list) else [],
                    complexity=description_data.get('complexity', 'medium'),
                    confidence=confidence,
                    processing_time=processing_time,
                    conversion_strategy=strategy_data if isinstance(strategy_data, dict) else {}
                )
                
            except Exception as e:
                logger.warning(f"LLM analysis attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    # Return fallback analysis
                    return self._get_fallback_analysis(oracle_sql, filename, time.time() - start_time)
        
        return self._get_fallback_analysis(oracle_sql, filename, time.time() - start_time)
    
    async def _call_llm_analysis(self, oracle_sql: str, prompt_config: Dict[str, Any]) -> str:
        """Call LLM for analysis with template rendering"""
        
        # Render prompt template
        template = Template(prompt_config["template"])
        rendered_prompt = template.render(oracle_sql=oracle_sql)
        
        # Prepare request
        request_data = {
            "model": os.getenv('LLM_MODEL_FAST', 'deepseek-1.3b-q5'),
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Oracle database analyst. Provide accurate, structured analysis in JSON format."
                },
                {
                    "role": "user", 
                    "content": rendered_prompt
                }
            ],
            "max_tokens": prompt_config.get("max_tokens", 500),
            "temperature": prompt_config.get("temperature", 0.1)
        }
        
        # Try each endpoint
        for endpoint in self.llm_endpoints:
            try:
                response = requests.post(
                    endpoint,
                    json=request_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                else:
                    logger.warning(f"LLM endpoint {endpoint} returned {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"LLM endpoint {endpoint} failed: {str(e)}")
        
        raise Exception("All LLM endpoints failed")
    
    def _parse_llm_json(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response as JSON with fallback"""
        try:
            # Try to extract JSON from response
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = llm_response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback: try parsing entire response
                return json.loads(llm_response)
                
        except json.JSONDecodeError:
            # Return structured fallback
            return {"error": "Failed to parse LLM response", "raw_response": llm_response}
    
    def _calculate_analysis_confidence(self, description: Dict, features: Any, strategy: Dict) -> float:
        """Calculate confidence score for LLM analysis"""
        confidence = 0.0
        
        # Check description quality
        if isinstance(description, dict) and 'description' in description:
            confidence += 0.3
            if len(description['description']) > 10:
                confidence += 0.1
        
        # Check features detection
        if isinstance(features, list) and len(features) > 0:
            confidence += 0.3
        
        # Check strategy quality
        if isinstance(strategy, dict) and len(strategy) > 0:
            confidence += 0.3
        
        return min(confidence, 1.0)
    
    def _get_fallback_analysis(self, oracle_sql: str, filename: str, processing_time: float) -> LLMAnalysisResult:
        """Get fallback analysis when LLM fails"""
        
        # Basic analysis without LLM
        features = []
        complexity = "medium"
        
        oracle_upper = oracle_sql.upper()
        if "BULK COLLECT" in oracle_upper:
            features.append("bulk_collect")
        if "FORALL" in oracle_upper:
            features.append("forall_statement")
        if "CURSOR" in oracle_upper:
            features.append("cursor_operations")
        if "EXCEPTION" in oracle_upper:
            features.append("exception_handling")
        
        # Determine complexity
        if len(oracle_sql) > 1000 or len(features) > 3:
            complexity = "high"
        elif len(oracle_sql) < 300 or len(features) < 2:
            complexity = "low"
        
        return LLMAnalysisResult(
            description=f"Oracle SQL file with {len(features)} detected features",
            purpose="Database operations and procedures",
            features=features,
            complexity=complexity,
            confidence=0.7,  # Moderate confidence for fallback
            processing_time=processing_time,
            conversion_strategy={"fallback": True, "manual_review_required": True}
        )
    
    def generate_dynamic_mcp(self, 
                           oracle_sql: str, 
                           filename: str, 
                           llm_analysis: Optional[LLMAnalysisResult] = None) -> str:
        """Generate MCP configuration dynamically using templates and LLM analysis"""
        
        start_time = time.time()
        
        # Generate file metadata
        file_metadata = FileMetadata(
            filename=filename,
            file_size=len(oracle_sql.encode('utf-8')),
            file_hash=hashlib.md5(oracle_sql.encode('utf-8')).hexdigest()[:12],
            upload_timestamp=datetime.now(),
            processing_time_ms=0  # Will be updated
        )
        
        # Use provided analysis or create fallback
        if llm_analysis is None:
            llm_analysis = self._get_fallback_analysis(oracle_sql, filename, 0)
        
        # Prepare template variables
        template_vars = {
            # File information
            "filename": filename,
            "file_id": file_metadata.file_hash,
            "file_size": file_metadata.file_size,
            "file_hash": file_metadata.file_hash,
            
            # LLM analysis results
            "llm_description": llm_analysis.description,
            "llm_complexity": llm_analysis.complexity,
            "llm_features": llm_analysis.features,
            "llm_conversion_strategy": llm_analysis.conversion_strategy,
            "analysis_confidence": llm_analysis.confidence,
            
            # Dynamic names and IDs
            "server_name": f"sakthi-oracle-{file_metadata.file_hash}",
            "resource_name": f"oracle_file_{file_metadata.file_hash}",
            
            # Timestamps
            "analysis_timestamp": datetime.now().isoformat(),
            "generation_timestamp": datetime.now().isoformat(),
            
            # Processing metadata
            "processing_time": llm_analysis.processing_time * 1000,  # Convert to ms
            "estimated_time": self._estimate_conversion_time(llm_analysis.complexity),
            "estimated_accuracy": self._estimate_accuracy(llm_analysis.confidence),
            "features_count": len(llm_analysis.features),
            
            # Technical details
            "llm_model": os.getenv('LLM_MODEL_FAST', 'deepseek-1.3b-q5'),
            "template_version": self.config["mcp_template_config"]["version"],
            "complexity_score": self._calculate_complexity_score(oracle_sql, llm_analysis.features),
            "manual_review_required": llm_analysis.confidence < 0.8,
            "automated_tests_available": self._has_automated_tests(llm_analysis.features),
            
            # Conversion mappings
            "data_type_mappings": self._generate_data_type_mappings(oracle_sql),
            "construct_mappings": self._generate_construct_mappings(llm_analysis.features),
            "optimization_hints": self._generate_optimization_hints(llm_analysis.conversion_strategy)
        }
        
        # Load and render template
        mcp_template = self.config["mcp_template_config"]["mcp_template"]
        template = Template(json.dumps(mcp_template, indent=2))
        rendered_mcp = template.render(**template_vars)
        
        # Update processing time
        total_time = (time.time() - start_time) * 1000
        file_metadata.processing_time_ms = total_time
        
        # Parse and return as formatted JSON
        mcp_config = json.loads(rendered_mcp)
        return json.dumps(mcp_config, indent=2)
    
    def _estimate_conversion_time(self, complexity: str) -> str:
        """Estimate conversion time based on complexity"""
        time_estimates = {
            "low": "5-10 minutes",
            "medium": "15-30 minutes", 
            "high": "45-90 minutes"
        }
        return time_estimates.get(complexity, "15-30 minutes")
    
    def _estimate_accuracy(self, confidence: float) -> str:
        """Estimate conversion accuracy based on confidence"""
        if confidence >= 0.9:
            return "95-98%"
        elif confidence >= 0.7:
            return "85-92%"
        else:
            return "70-85%"
    
    def _calculate_complexity_score(self, oracle_sql: str, features: List[str]) -> float:
        """Calculate numerical complexity score"""
        base_score = len(oracle_sql) / 1000  # Base on code length
        feature_bonus = len(features) * 0.2  # Bonus for features
        return min(base_score + feature_bonus, 10.0)
    
    def _has_automated_tests(self, features: List[str]) -> bool:
        """Check if automated tests are available for detected features"""
        testable_features = ["bulk_collect", "forall_statement", "cursor_operations"]
        return any(feature in testable_features for feature in features)
    
    def _generate_data_type_mappings(self, oracle_sql: str) -> Dict[str, str]:
        """Generate data type mappings based on detected types"""
        mappings = {}
        
        if "PLS_INTEGER" in oracle_sql.upper():
            mappings["PLS_INTEGER"] = "INT64"
        if "%TYPE" in oracle_sql.upper():
            mappings["anchored_types"] = "Use appropriate BigQuery types"
        if "VARCHAR2" in oracle_sql.upper():
            mappings["VARCHAR2"] = "STRING"
        if "NUMBER" in oracle_sql.upper():
            mappings["NUMBER"] = "NUMERIC"
        if "BOOLEAN" in oracle_sql.upper():
            mappings["BOOLEAN"] = "BOOL"
            
        return mappings
    
    def _generate_construct_mappings(self, features: List[str]) -> Dict[str, str]:
        """Generate construct mappings based on detected features"""
        mappings = {}
        
        feature_map = {
            "bulk_collect": "Use ARRAY operations and batch processing",
            "forall_statement": "Convert to BigQuery DML with UNNEST",
            "cursor_operations": "Use BigQuery loops or set-based operations",
            "exception_handling": "Use BigQuery EXCEPTION blocks",
            "dbms_output": "Convert to SELECT statements or logging"
        }
        
        for feature in features:
            if feature in feature_map:
                mappings[feature] = feature_map[feature]
        
        return mappings
    
    def _generate_optimization_hints(self, conversion_strategy: Dict[str, Any]) -> List[str]:
        """Generate optimization hints based on conversion strategy"""
        hints = [
            "Consider partitioning large tables for better performance",
            "Use ARRAY operations instead of row-by-row processing",
            "Leverage BigQuery's columnar storage for analytics"
        ]
        
        if conversion_strategy.get("bulk_operations"):
            hints.append("Optimize bulk operations using BigQuery's native batch processing")
        
        if conversion_strategy.get("complex_joins"):
            hints.append("Consider denormalizing data for better BigQuery performance")
        
        return hints
    
    def validate_mcp_config(self, mcp_config: str) -> Dict[str, Any]:
        """Validate generated MCP configuration"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "score": 0.0
        }
        
        try:
            # Parse JSON
            config = json.loads(mcp_config)
            
            # Check required fields
            required_fields = ["mcpVersion", "server", "capabilities", "resources", "tools", "prompts"]
            for field in required_fields:
                if field not in config:
                    validation_result["errors"].append(f"Missing required field: {field}")
                    validation_result["valid"] = False
            
            # Check MCP version
            if config.get("mcpVersion") != "2024-11-05":
                validation_result["warnings"].append("MCP version may not be latest")
            
            # Validate resources
            resources = config.get("resources", [])
            if not resources:
                validation_result["warnings"].append("No resources defined")
            else:
                for i, resource in enumerate(resources):
                    if "uri" not in resource:
                        validation_result["errors"].append(f"Resource {i} missing URI")
                        validation_result["valid"] = False
            
            # Validate tools
            tools = config.get("tools", [])
            for i, tool in enumerate(tools):
                if "name" not in tool or "inputSchema" not in tool:
                    validation_result["errors"].append(f"Tool {i} missing required fields")
                    validation_result["valid"] = False
            
            # Calculate score
            total_checks = 10
            passed_checks = total_checks - len(validation_result["errors"])
            validation_result["score"] = (passed_checks / total_checks) * 100
            
        except json.JSONDecodeError as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Invalid JSON: {str(e)}")
            validation_result["score"] = 0.0
        
        return validation_result
    
    def save_mcp_template(self, mcp_config: str, filename: str, output_dir: str = "generated_mcp") -> str:
        """Save MCP configuration as downloadable template"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(filename).stem
        output_filename = f"{base_name}_mcp_{timestamp}.json"
        output_file = output_path / output_filename
        
        # Save with metadata
        template_data = {
            "metadata": {
                "original_file": filename,
                "generated_at": datetime.now().isoformat(),
                "generator": "sakthi-dynamic-mcp-engine",
                "version": "1.0.0"
            },
            "mcp_config": json.loads(mcp_config)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, indent=2)
        
        logger.info(f"MCP template saved: {output_file}")
        return str(output_file)

# Integration class for Sakthi Platform
class SakthiMCPIntegration:
    """Integration class for Sakthi Platform with dynamic MCP generation"""
    
    def __init__(self, sakthi_engine, config_path: str = "mcp_templates/oracle_bigquery_config.json"):
        self.sakthi_engine = sakthi_engine
        self.mcp_engine = DynamicMCPEngine(config_path)
    
    async def process_oracle_file_to_mcp(self, 
                                       oracle_sql: str, 
                                       filename: str,
                                       include_llm_analysis: bool = True) -> Dict[str, Any]:
        """Process Oracle file and generate comprehensive MCP configuration"""
        
        logger.info(f"Processing Oracle file: {filename}")
        
        result = {
            "success": False,
            "filename": filename,
            "mcp_config": None,
            "llm_analysis": None,
            "validation": None,
            "download_path": None,
            "processing_time": 0,
            "error": None
        }
        
        start_time = time.time()
        
        try:
            # Step 1: LLM Analysis (if enabled)
            llm_analysis = None
            if include_llm_analysis:
                logger.info("Starting LLM analysis...")
                llm_analysis = await self.mcp_engine.analyze_oracle_file_with_llm(oracle_sql, filename)
                result["llm_analysis"] = {
                    "description": llm_analysis.description,
                    "features": llm_analysis.features,
                    "complexity": llm_analysis.complexity,
                    "confidence": llm_analysis.confidence,
                    "conversion_strategy": llm_analysis.conversion_strategy
                }
                logger.info(f"LLM analysis completed with confidence: {llm_analysis.confidence:.2f}")
            
            # Step 2: Generate dynamic MCP configuration
            logger.info("Generating dynamic MCP configuration...")
            mcp_config = self.mcp_engine.generate_dynamic_mcp(oracle_sql, filename, llm_analysis)
            result["mcp_config"] = mcp_config
            
            # Step 3: Validate MCP configuration
            logger.info("Validating MCP configuration...")
            validation = self.mcp_engine.validate_mcp_config(mcp_config)
            result["validation"] = validation
            
            if not validation["valid"]:
                logger.warning(f"MCP validation issues: {validation['errors']}")
            
            # Step 4: Save as downloadable template
            logger.info("Saving MCP template...")
            download_path = self.mcp_engine.save_mcp_template(mcp_config, filename)
            result["download_path"] = download_path
            
            # Step 5: Update processing metadata
            result["processing_time"] = time.time() - start_time
            result["success"] = True
            
            logger.info(f"Successfully processed {filename} in {result['processing_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            result["error"] = str(e)
            result["processing_time"] = time.time() - start_time
        
        return result
    
    def get_processing_status(self, file_hash: str) -> Dict[str, Any]:
        """Get processing status for a file"""
        # This could be enhanced with a database or cache
        return {
            "file_hash": file_hash,
            "status": "unknown",
            "message": "Status tracking not implemented yet"
        }

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_dynamic_mcp():
        """Test the dynamic MCP generation"""
        
        # Sample Oracle SQL
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
        
        # Initialize engine
        mcp_engine = DynamicMCPEngine()
        
        # Test LLM analysis
        print("=== Testing LLM Analysis ===")
        llm_analysis = await mcp_engine.analyze_oracle_file_with_llm(oracle_sql, "test_procedure.sql")
        print(f"Description: {llm_analysis.description}")
        print(f"Features: {llm_analysis.features}")
        print(f"Complexity: {llm_analysis.complexity}")
        print(f"Confidence: {llm_analysis.confidence:.2f}")
        
        # Test MCP generation
        print("\n=== Testing MCP Generation ===")
        mcp_config = mcp_engine.generate_dynamic_mcp(oracle_sql, "test_procedure.sql", llm_analysis)
        print("MCP Config generated successfully!")
        
        # Test validation
        print("\n=== Testing Validation ===")
        validation = mcp_engine.validate_mcp_config(mcp_config)
        print(f"Valid: {validation['valid']}")
        print(f"Score: {validation['score']:.1f}%")
        if validation['errors']:
            print(f"Errors: {validation['errors']}")
        
        # Save template
        print("\n=== Saving Template ===")
        template_path = mcp_engine.save_mcp_template(mcp_config, "test_procedure.sql")
        print(f"Template saved: {template_path}")
        
        print("\n=== Test Complete ===")
    
    # Run test
    asyncio.run(test_dynamic_mcp())