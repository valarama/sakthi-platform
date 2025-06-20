import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from core import SakthiEngine


# Test MCP generation
sakthi = SakthiEngine()

# Test with your Oracle SQL
input_text = "Convert Oracle schema to BigQuery with data type optimization"

# Generate MCP config
mcp_config = sakthi.generate_mcp_config(input_text)

# Save result
with open("oracle_mcp_config.json", "w") as f:
    f.write(mcp_config)

print("✅ Generated MCP config: oracle_mcp_config.json")
print(mcp_config[:500] + "...")