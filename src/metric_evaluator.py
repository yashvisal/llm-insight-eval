"""
Simple metric evaluation logic - sequential evaluation
"""
import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from .llm_client import OllamaClient
from .models import MetricEvaluation

class MetricEvaluator:
    """Handles metric evaluation logic using simple sequential calls"""
    
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client
        
        # Simple metric descriptions
        self.metrics = {
            "correctness": "Evaluate if the claim is factually correct based on the data analysis. Score 1-5 where 1=completely incorrect, 5=completely correct.",
            "helpfulness": "Evaluate if the claim provides useful insights. Score 1-5 where 1=not helpful, 5=very helpful.",
            "complexity": "Evaluate the complexity of the claim. Score 1-5 where 1=very simple, 5=very complex.",
            "coherence": "Evaluate if the claim is logically coherent and well-structured. Score 1-5 where 1=incoherent, 5=highly coherent.",
            "verbosity": "Evaluate the verbosity of the claim. Score 1-5 where 1=too brief, 5=too verbose."
        }
    
    async def evaluate_all_metrics(
        self,
        claim: str,
        dataset_info: str,
        analysis_results: str
    ) -> Dict[str, MetricEvaluation]:
        """Evaluate all metrics sequentially"""
        
        logger.info("Evaluating all metrics sequentially...")
        
        metric_results = {}
        
        for metric_name, metric_description in self.metrics.items():
            try:
                logger.info(f"Evaluating {metric_name}...")
                
                # Create simple prompt
                system_prompt = f"""You are an expert evaluator. Evaluate the given claim based on the data analysis results.

{metric_description}

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

                # Send to LLM
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                response = await self.llm_client.ainvoke(messages)
                
                # Try to parse JSON response
                try:
                    # Clean up response - remove markdown if present
                    clean_response = response.strip()
                    if clean_response.startswith("```json"):
                        clean_response = clean_response[7:]
                    if clean_response.endswith("```"):
                        clean_response = clean_response[:-3]
                    
                    result = json.loads(clean_response.strip())
                    
                    # Validate required fields
                    if "score" not in result or "rationale" not in result:
                        raise ValueError("Missing required fields")
                    
                    metric_results[metric_name] = MetricEvaluation(
                        score=result["score"],
                        rationale=result["rationale"],
                        metric_name=metric_name
                    )
                    
                    logger.info(f"✅ {metric_name} evaluated: score {result['score']}")
                    
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error parsing {metric_name} response: {str(e)}")
                    logger.error(f"Raw response: {response}")
                    
                    # Create fallback result
                    metric_results[metric_name] = MetricEvaluation(
                        score=1,
                        rationale=f"Error parsing response: {str(e)}",
                        metric_name=metric_name
                    )
                
            except Exception as e:
                logger.error(f"Error evaluating {metric_name}: {str(e)}")
                metric_results[metric_name] = MetricEvaluation(
                    score=1,
                    rationale=f"Error in evaluation: {str(e)}",
                    metric_name=metric_name
                )
        
        logger.info("Sequential metrics evaluation completed")
        return metric_results
    
    async def evaluate_metric(
        self, 
        metric_name: str, 
        claim: str, 
        dataset_info: str,
        analysis_results: str
    ) -> MetricEvaluation:
        """Evaluate a single metric"""
        
        if metric_name not in self.metrics:
            return MetricEvaluation(
                score=1,
                rationale=f"Unknown metric: {metric_name}",
                metric_name=metric_name
            )
        
        # Use the sequential evaluator for single metric
        results = await self.evaluate_all_metrics(claim, dataset_info, analysis_results)
        return results.get(metric_name, MetricEvaluation(
            score=1,
            rationale=f"Metric {metric_name} not found",
            metric_name=metric_name
        )) 