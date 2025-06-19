"""
Basic test to verify E2B is working correctly
"""
import asyncio
from loguru import logger
from e2b_code_interpreter import Sandbox

async def test_e2b_basic():
    """Test basic E2B functionality"""
    
    logger.info("Testing basic E2B functionality...")
    
    try:
        with Sandbox() as sandbox:
            # Test simple code execution
            logger.info("Running simple Python code...")
            execution = sandbox.run_code("print('Hello from E2B!')")
            print(f"Output: {execution.logs.stdout}")
            
            # Test pandas import and basic operations
            logger.info("Testing pandas operations...")
            pandas_code = """
import pandas as pd
import numpy as np

# Create a simple dataframe
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': ['a', 'b', 'c', 'd', 'e']
})

print("DataFrame created successfully:")
print(df)
print(f"Shape: {df.shape}")
"""
            execution = sandbox.run_code(pandas_code)
            print(f"Pandas test output:\n{execution.logs.stdout}")
            
            if execution.logs.stderr:
                print(f"Errors: {execution.logs.stderr}")
            
        logger.info("✅ Basic E2B test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Basic E2B test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_e2b_basic()) 