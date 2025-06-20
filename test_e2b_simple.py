"""
Simplified test to debug E2B integration
"""
import asyncio
import json
from loguru import logger
from e2b_code_interpreter import Sandbox

async def test_e2b_simple():
    """Test E2B with a simple data analysis task"""
    
    logger.info("Testing E2B with simple data analysis...")
    
    try:
        with Sandbox() as sandbox:
            # Upload the dataset
            with open("data/train.csv", 'r') as f:
                dataset_content = f.read()
            
            sandbox.files.write("train.csv", dataset_content)
            logger.info("✅ Dataset uploaded successfully")
            
            # Simple analysis code
            analysis_code = """
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('train.csv')

# Basic analysis for the claim about average item weight
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Check if Item_Weight column exists
if 'Item_Weight' in df.columns:
    avg_weight = df['Item_Weight'].mean()
    print(f"Average item weight: {avg_weight:.2f} kg")
    print(f"Min weight: {df['Item_Weight'].min():.2f} kg")
    print(f"Max weight: {df['Item_Weight'].max():.2f} kg")
    print(f"Standard deviation: {df['Item_Weight'].std():.2f} kg")
else:
    print("Item_Weight column not found")
    print("Available columns:", df.columns.tolist())
"""
            
            # Execute the code
            execution = sandbox.run_code(analysis_code)
            
            # Print results
            print("\n=== E2B Execution Results ===")
            print("STDOUT:")
            if execution.logs.stdout:
                for line in execution.logs.stdout:
                    print(f"  {line}")
            
            print("\nSTDERR:")
            if execution.logs.stderr:
                for line in execution.logs.stderr:
                    print(f"  {line}")
            
            print(f"\nSuccess: {not bool(execution.logs.stderr)}")
            
        logger.info("✅ E2B test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ E2B test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_e2b_simple()) 