import os

def create_project_structure(base_dir, folders, packages):
    print(f"Starting project structure creation in: {base_dir}\n")
    
    # Create directories
    for folder in folders:
        path = os.path.join(base_dir, folder)
        os.makedirs(path, exist_ok=True)
        print(f"✔️ Created directory: {path}")
    
    # Create __init__.py files
    for package in packages:
        init_path = os.path.join(base_dir, package, "__init__.py")
        os.makedirs(os.path.dirname(init_path), exist_ok=True)  # Ensure the directory exists
        with open(init_path, "w") as f:
            f.write("# Initialization file for " + package)
        print(f"✔️ Created file: {init_path}")
    
    print("\n🎉 Project structure created successfully!")

# Use the current working directory as the base directory
base_dir = os.getcwd()

# Define folder structure
folders = [
    "backend/api",
    "sakthi-language",
    "document-processor",
    "genai-modeling-agent",
    "sakthi-llm-integration",
    "web-interface/pages",
    "web-interface/components",
    "web-interface/styles",
    "deployment/kubernetes",
    "config",
    "docs",
    "tests",
    "logs",
    "uploads",
    "storage",
    "chromadb"
]

# Define packages for __init__.py
packages = [
    "backend",
    "backend/api",
    "sakthi-language",
    "document-processor",
    "genai-modeling-agent",
    "sakthi-llm-integration",
    "tests"
]

# Create structure
create_project_structure(base_dir, folders, packages)
# This script creates a project structure for the Sakthi platform with specified folders and packages.
# It initializes directories and creates __init__.py files for Python packages.