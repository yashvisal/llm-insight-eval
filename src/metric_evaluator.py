"""
Metric evaluation logic
"""
import asyncio
import json
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from .llm_client import OllamaClient
from .models import MetricEvaluation

class MetricEvaluator:
    """Handles metric evaluation logic"""
    
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client
        
        self.metric_prompts = {
            "correctness": "Evaluate if the claim is factually correct based on the data analysis. Score 1-5 where 1=completely incorrect, 5=completely correct.",
            "helpfulness": "Evaluate if the claim provides useful insights. Score 1-5 where 1=not helpful, 5=very helpful.",
            "complexity": "Evaluate the complexity of the claim. Score 1-5 where 1=very simple, 5=very complex.",
            "coherence": "Evaluate if the claim is logically coherent and well-structured. Score 1-5 where 1=incoherent, 5=highly coherent.",
            "verbosity": "Evaluate the verbosity of the claim. Score 1-5 where 1=too brief, 5=too verbose."
        }
    
    async def evaluate_metric(
        self, 
        metric_name: str, 
        claim: str, 
        dataset_info: str,
        analysis_results: str
    ) -> MetricEvaluation:
        """Evaluate a single metric for a given claim"""
        
        system_prompt = f"""You are an expert evaluator. Evaluate the given claim based on the data analysis results.

{self.metric_prompts[metric_name]}

Respond with a JSON object containing:
- "score": integer from 1-5
- "rationale": string explaining your reasoning

Example: {{"score": 4, "rationale": "The claim is mostly correct because..."}}"""
        
        user_prompt = f"""
Dataset Info: {dataset_info}

Claim: {claim}

Data Analysis Results:
{analysis_results}

Evaluate the {metric_name} of this claim.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        logger.info(f"Evaluating {metric_name}...")
        response = await self.llm_client.ainvoke(messages)
        
        # Parse JSON response
        try:
            # Extract JSON from response
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                result = json.loads(json_str)
                
                # Validate required fields
                if "score" not in result or "rationale" not in result:
                    raise ValueError("Missing required fields")
                
                return MetricEvaluation(
                    score=result["score"],
                    rationale=result["rationale"],
                    metric_name=metric_name
                )
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.error(f"Error parsing {metric_name} response: {str(e)}")
            return MetricEvaluation(
                score=1,
                rationale=f"Error parsing response: {str(e)}",
                metric_name=metric_name
            )
    
    async def evaluate_all_metrics(
        self,
        claim: str,
        dataset_info: str,
        analysis_results: str
    ) -> Dict[str, MetricEvaluation]:
        """Evaluate all metrics in parallel"""
        
        metrics = ["correctness", "helpfulness", "complexity", "coherence", "verbosity"]
        
        # Create tasks for parallel evaluation
        tasks = []
        for metric in metrics:
            task = self.evaluate_metric(metric, claim, dataset_info, analysis_results)
            tasks.append(task)
        
        # Wait for all evaluations to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        metric_results = {}
        for i, metric in enumerate(metrics):
            if isinstance(results[i], Exception):
                logger.error(f"Error evaluating {metric}: {str(results[i])}")
                metric_results[metric] = MetricEvaluation(
                    score=1,
                    rationale=f"Error: {str(results[i])}",
                    metric_name=metric
                )
            else:
                metric_results[metric] = results[i]
        
        return metric_results 