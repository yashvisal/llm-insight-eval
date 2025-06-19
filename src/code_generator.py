"""
Code generation logic for data analysis
"""
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from .llm_client import OllamaClient

class CodeGenerator:
    """Handles code generation for data analysis"""
    
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client
    
    async def generate_analysis_code(self, claim: str, dataset_info: str) -> str:
        """Generate Python code for analyzing a specific claim"""
        
        system_prompt = """You are a data scientist. Your task is to generate Python code to analyze a dataset and test a specific claim.

Available libraries: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn

The code should:
1. Load the dataset (assume it's already uploaded as 'dataset.csv')
2. Perform analysis relevant to the claim
3. Generate visualizations if helpful
4. Print clear results that can be used to evaluate the claim

Return ONLY the Python code, no explanations."""
        
        user_prompt = f"""
Dataset Info: {dataset_info}

Claim to analyze: {claim}

Generate Python code to test this claim. Focus on statistical analysis and visualizations that would help determine if the claim is true or false.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        logger.info("Generating analysis code...")
        code = await self.llm_client.generate_response(messages)
        
        # Clean up the response (remove markdown if present)
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        logger.info(f"Generated code length: {len(code)} characters")
        return code.strip() 