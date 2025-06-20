import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from core import SakthiEngine
from mcp.generators.mcp_generator import SakthiMCPGenerator

# Your Oracle SQL processing
with open("06_ORALL__BULKCOLLECT EXAMPLE.sql", "r") as f:
    oracle_sql = f.read()

sakthi_engine = SakthiEngine()
intent = sakthi_engine.parser.parse(f"Convert Oracle schema to BigQuery: {oracle_sql}")

mcp_generator = SakthiMCPGenerator()
mcp_server = mcp_generator.generate_mcp_from_intent(intent)
mcp_json = mcp_generator.export_mcp_json(mcp_server)

with open("oracle_to_bigquery_mcp.json", "w") as f:
    f.write(mcp_json)

print("✅ Generated MCP config: oracle_to_bigquery_mcp.json")