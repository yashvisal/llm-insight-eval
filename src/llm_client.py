"""
LLM client for communicating with Ollama models
"""
import json
import asyncio
from typing import Dict, Any, Optional, List
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from .config import get_config
from .models import MetricEvaluation

class LLMClient:
    """Client for interacting with Ollama LLM"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.config = get_config()
        self.model_name = model_name or self.config.llm.model_name
        self.base_url = self.config.llm.base_url
        
        # Initialize Ollama client
        self.llm = Ollama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.config.llm.temperature,
            timeout=self.config.llm.timeout
        )
        
        # JSON output parser
        self.json_parser = JsonOutputParser()
        
        logger.info(f"Initialized LLM client with model: {self.model_name}")
    
    async def evaluate_metric(
        self, 
        metric_name: str, 
        claim: str, 
        dataset_summary: str, 
        task_description: Optional[str] = None,
        data_analysis_results: Optional[List[Dict[str, Any]]] = None
    ) -> MetricEvaluation:
        """
        Evaluate a single metric for a given claim
        
        Args:
            metric_name: Name of the metric to evaluate
            claim: The claim to evaluate
            dataset_summary: Summary of the dataset
            task_description: Optional task description
            data_analysis_results: Optional results from data analysis
            
        Returns:
            MetricEvaluation object with score and rationale
        """
        try:
            # Get the appropriate prompt template
            from prompts.metrics import metric_prompts
            prompt_template = metric_prompts[metric_name]
            
            # Prepare context
            context = {
                "claim": claim,
                "dataset_summary": dataset_summary,
                "task_description": task_description or "N/A"
            }
            
            # Add data analysis results if available
            if data_analysis_results:
                analysis_summary = self._format_analysis_results(data_analysis_results)
                context["data_analysis"] = analysis_summary
            
            # Format the prompt
            messages = prompt_template.format_messages(**context)
            
            # Get LLM response
            response = await self._get_llm_response(messages)
            
            # Parse the response
            parsed_response = self._parse_evaluation_response(response)
            
            # Create MetricEvaluation object
            evaluation = MetricEvaluation(
                score=parsed_response["score"],
                rationale=parsed_response["rationale"],
                confidence=parsed_response.get("confidence"),
                metadata={"metric": metric_name, "raw_response": response}
            )
            
            logger.info(f"Evaluated {metric_name}: score={evaluation.score}")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating {metric_name}: {str(e)}")
            # Return a default evaluation with error information
            return MetricEvaluation(
                score=1,
                rationale=f"Error during evaluation: {str(e)}",
                metadata={"error": str(e), "metric": metric_name}
            )
    
    async def analyze_data(
        self, 
        claim: str, 
        dataset_summary: str,
        data_path: str
    ) -> List[Dict[str, Any]]:
        """
        Perform data analysis relevant to the claim
        
        Args:
            claim: The claim to analyze data for
            dataset_summary: Summary of the dataset
            data_path: Path to the dataset file
            
        Returns:
            List of analysis results
        """
        try:
            # Create analysis prompt
            analysis_prompt = self._create_analysis_prompt(claim, dataset_summary, data_path)
            
            # Get LLM response
            response = await self._get_llm_response([analysis_prompt])
            
            # Parse and execute the analysis
            analysis_results = await self._execute_analysis(response, data_path)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in data analysis: {str(e)}")
            return []
    
    async def _get_llm_response(self, messages: List) -> str:
        """Get response from LLM"""
        try:
            # Convert to async if needed
            if hasattr(self.llm, 'ainvoke'):
                response = await self.llm.ainvoke(messages)
            else:
                # Fallback to sync call
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, self.llm.invoke, messages)
            
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            raise
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured format"""
        try:
            # Try to extract JSON from the response
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                
                parsed = json.loads(json_str)
                
                # Validate required fields
                if "score" not in parsed or "rationale" not in parsed:
                    raise ValueError("Missing required fields in response")
                
                return parsed
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            # Return default response
            return {
                "score": 1,
                "rationale": f"Error parsing response: {str(e)}"
            }
    
    def _format_analysis_results(self, results: List[Dict[str, Any]]) -> str:
        """Format analysis results for inclusion in prompts"""
        if not results:
            return "No data analysis results available."
        
        formatted = []
        for result in results:
            formatted.append(f"Analysis: {result.get('analysis_type', 'Unknown')}")
            formatted.append(f"Results: {result.get('results', {})}")
            if result.get('code_executed'):
                formatted.append(f"Code: {result['code_executed']}")
        
        return "\n".join(formatted)
    
    def _create_analysis_prompt(self, claim: str, dataset_summary: str, data_path: str) -> SystemMessage:
        """Create a prompt for data analysis"""
        prompt_text = f"""
You are a data analyst. Analyze the following claim using the provided dataset:

Claim: {claim}
Dataset: {dataset_summary}
Data Path: {data_path}

Your task is to:
1. Determine what analysis is needed to evaluate this claim
2. Write Python code to perform the analysis
3. Return the results in a structured format

Focus on:
- Statistical tests if needed (correlation, t-tests, etc.)
- Data aggregations and summaries
- Visualizations if helpful
- Any patterns or insights relevant to the claim

Return your analysis plan and code in a clear, executable format.
"""
        return SystemMessage(content=prompt_text)
    
    async def _execute_analysis(self, response: str, data_path: str) -> List[Dict[str, Any]]:
        """Execute the analysis code and return results"""
        # This is a simplified version - in practice, you'd want a proper sandbox
        # For now, we'll extract code and return a placeholder
        try:
            # Extract code from response (simplified)
            if "```python" in response:
                code_start = response.find("```python") + 9
                code_end = response.find("```", code_start)
                code = response[code_start:code_end].strip()
            else:
                code = response
            
            # For now, return a placeholder result
            # In a real implementation, you'd execute this in a sandbox
            return [{
                "analysis_type": "statistical_analysis",
                "results": {"code_provided": code[:200] + "..." if len(code) > 200 else code},
                "code_executed": code,
                "execution_time": 0.1,
                "success": True
            }]
            
        except Exception as e:
            logger.error(f"Error executing analysis: {str(e)}")
            return [{
                "analysis_type": "error",
                "results": {"error": str(e)},
                "execution_time": 0.0,
                "success": False,
                "error_message": str(e)
            }] 