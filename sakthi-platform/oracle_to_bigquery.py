"""
Oracle to BigQuery Conversion Script
Uses Sakthi Platform for intelligent schema transformation
"""

import os
import sys
import glob
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append('.')

def convert_oracle_file(oracle_file_path, output_dir="bigquery-output"):
    """Convert a single Oracle SQL file to BigQuery"""
    
    print(f"🔄 Converting: {oracle_file_path}")
    
    try:
        # Read Oracle SQL file
        with open(oracle_file_path, 'r', encoding='utf-8') as f:
            oracle_sql = f.read()
        
        # Import Sakthi after path setup
        from core import SakthiEngine, OutputFormat
        
        # Initialize Sakthi engine
        sakthi = SakthiEngine()
        
        # Create conversion intent
        intent = """Convert Oracle schema to BigQuery format:
        1. Convert Oracle data types to BigQuery equivalents
        2. Handle constraints appropriately 
        3. Maintain table relationships
        4. Add project and dataset references
        5. Include performance optimizations"""
        
        # Combine intent with Oracle SQL
        input_text = f"{intent}\n\nOracle SQL to convert:\n{oracle_sql}"
        
        # Process with Sakthi
        result = sakthi.process(input_text, OutputFormat.SQL)
        
        # Create output filename
        file_name = Path(oracle_file_path).stem
        output_file = f"{output_dir}/{file_name}_bigquery.sql"
        
        # Ensure output directory exists
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save BigQuery SQL
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"-- Converted from Oracle: {oracle_file_path}\n")
            f.write(f"-- Conversion confidence: {result.confidence:.2f}\n")
            f.write(f"-- Generated on: {result.metadata.get('timestamp', 'N/A')}\n\n")
            f.write(result.content)
        
        print(f"✅ Converted successfully!")
        print(f"📁 Output: {output_file}")
        print(f"🎯 Confidence: {result.confidence:.2f}")
        print("-" * 50)
        
        return {
            'success': True,
            'input_file': oracle_file_path,
            'output_file': output_file,
            'confidence': result.confidence,
            'content': result.content
        }
        
    except Exception as e:
        print(f"❌ Error converting {oracle_file_path}: {str(e)}")
        return {
            'success': False,
            'input_file': oracle_file_path,
            'error': str(e)
        }

def convert_all_oracle_files(input_dir="oracle-files"):
    """Convert all Oracle SQL files in the input directory"""
    
    print("🚀 Starting batch Oracle to BigQuery conversion...")
    print(f"📂 Input directory: {input_dir}")
    print("=" * 60)
    
    # Find all SQL files
    sql_files = glob.glob(f"{input_dir}/*.sql")
    
    if not sql_files:
        print(f"❌ No SQL files found in {input_dir}")
        print("💡 Copy your Oracle SQL files to the oracle-files folder")
        return
    
    results = []
    
    for sql_file in sql_files:
        result = convert_oracle_file(sql_file)
        results.append(result)
    
    # Summary
    successful = len([r for r in results if r['success']])
    total = len(results)
    
    print("\n" + "=" * 60)
    print("📊 CONVERSION SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")
    
    if successful > 0:
        avg_confidence = sum(r.get('confidence', 0) for r in results if r['success']) / successful
        print(f"🎯 Average Confidence: {avg_confidence:.2f}")
        print(f"📁 Output files in: bigquery-output/")
    
    return results

if __name__ == "__main__":
    # Convert all Oracle files
    results = convert_all_oracle_files()
    
    # Optional: Convert specific file
    # result = convert_oracle_file("oracle-files/your_schema.sql")