import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=== Sakthi Platform Configuration Test ===")
print(f"LLM Provider: {os.getenv('LLM_PROVIDER')}")
print(f"ChromaDB Path: {os.getenv('CHROMADB_PATH')}")
print(f"Upload Path: {os.getenv('UPLOAD_PATH', './uploads')}")
print(f"Storage Path: {os.getenv('STORAGE_PATH', './storage')}")
print(f"SerpAPI Key: {os.getenv('SERPAPI_KEY')[:20]}...")
print(f"API Port: {os.getenv('API_PORT')}")

# Test folder creation
import pathlib
chromadb_path = os.getenv('CHROMADB_PATH', './chromadb')
pathlib.Path(chromadb_path).mkdir(parents=True, exist_ok=True)
print(f"✅ ChromaDB folder created: {chromadb_path}")

uploads_path = os.getenv('UPLOAD_PATH', './uploads')
pathlib.Path(uploads_path).mkdir(parents=True, exist_ok=True)
print(f"✅ Uploads folder created: {uploads_path}")

print("✅ Configuration test complete!")