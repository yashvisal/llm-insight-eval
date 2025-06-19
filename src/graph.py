"""
LangGraph workflow for LLM Insight Evaluation Agent
"""
import asyncio
import time
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from loguru import logger

from .models import EvaluationState, EvaluationScores, MetricEvaluation, DataAnalysisResult
from .llm_client import LLMClient
from .config import get_config
from .e2b_client import E2BClient

class EvaluationGraph:
    """Main evaluation workflow using LangGraph"""
    
    def __init__(self):
        self.config = get_config()
        self.llm_client = LLMClient()
        self.e2b_client = E2BClient()
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(EvaluationState)
        
        # Add nodes
        workflow.add_node("start", self._start_node)
        workflow.add_node("data_analysis", self._data_analysis_node)
        workflow.add_node("evaluate_metrics", self._evaluate_metrics_node)
        workflow.add_node("output", self._output_node)
        
        # Define edges
        workflow.set_entry_point("start")
        workflow.add_edge("start", "data_analysis")
        workflow.add_edge("data_analysis", "evaluate_metrics")
        workflow.add_edge("evaluate_metrics", "output")
        workflow.add_edge("output", END)
        
        return workflow.compile()
    
    async def _start_node(self, state: EvaluationState) -> EvaluationState:
        """Initial node that validates input and prepares for analysis"""
        logger.info("Starting evaluation workflow")
        
        try:
            # Validate inputs
            if not state.claim or not state.dataset_summary:
                raise ValueError("Claim and dataset summary are required")
            
            # Add basic dataset summary if not provided
            if not state.dataset_summary or state.dataset_summary == "":
                state.dataset_summary = self._get_default_dataset_summary()
            
            logger.info(f"Evaluating claim: {state.claim[:100]}...")
            return state
            
        except Exception as e:
            logger.error(f"Error in start node: {str(e)}")
            state.errors.append(f"Start node error: {str(e)}")
            return state
    
    async def _data_analysis_node(self, state: EvaluationState) -> EvaluationState:
        """Node that performs data analysis relevant to the claim using E2B"""
        logger.info("Performing data analysis with E2B")
        
        try:
            start_time = time.time()
            
            # Check if E2B is available
            if self.e2b_client.tool:
                logger.info("Using E2B for data analysis")
                
                # Upload dataset to E2B
                await self.e2b_client.upload_dataset(self.config.dataset.data_path)
                
                # Perform analysis using E2B
                analysis_results = await self.e2b_client.analyze_claim(
                    claim=state.claim,
                    dataset_summary=state.dataset_summary
                )
            else:
                logger.info("E2B not available, using fallback analysis")
                # Use existing LLM-based analysis
                analysis_results = await self.llm_client.analyze_data(
                    claim=state.claim,
                    dataset_summary=state.dataset_summary,
                    data_path=self.config.dataset.data_path
                )
            
            # Convert to DataAnalysisResult objects
            for result in analysis_results:
                data_result = DataAnalysisResult(
                    analysis_type=result.get("analysis_type", "unknown"),
                    results=result.get("results", {}),
                    code_executed=result.get("code_executed"),
                    execution_time=result.get("execution_time", 0.0),
                    success=result.get("success", False),
                    error_message=result.get("error_message")
                )
                state.data_analysis_results.append(data_result)
            
            analysis_time = time.time() - start_time
            logger.info(f"Data analysis completed in {analysis_time:.2f}s")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in data analysis node: {str(e)}")
            state.errors.append(f"Data analysis error: {str(e)}")
            return state
    
    async def _evaluate_metrics_node(self, state: EvaluationState) -> EvaluationState:
        """Node that evaluates all metrics in parallel"""
        logger.info("Evaluating metrics")
        
        try:
            start_time = time.time()
            
            # Prepare analysis results for metrics
            analysis_summary = self._format_analysis_for_metrics(state.data_analysis_results)
            
            # Evaluate all metrics
            if self.config.evaluation.parallel_evaluation:
                # Parallel evaluation
                tasks = []
                for metric in self.config.evaluation.metrics:
                    task = self.llm_client.evaluate_metric(
                        metric_name=metric,
                        claim=state.claim,
                        dataset_summary=state.dataset_summary,
                        task_description=state.task_description,
                        data_analysis_results=analysis_summary
                    )
                    tasks.append(task)
                
                # Wait for all evaluations to complete
                evaluations = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                metric_results = {}
                for i, metric in enumerate(self.config.evaluation.metrics):
                    if isinstance(evaluations[i], Exception):
                        logger.error(f"Error evaluating {metric}: {str(evaluations[i])}")
                        metric_results[metric] = MetricEvaluation(
                            score=1,
                            rationale=f"Error: {str(evaluations[i])}",
                            metadata={"error": str(evaluations[i])}
                        )
                    else:
                        metric_results[metric] = evaluations[i]
            else:
                # Sequential evaluation
                metric_results = {}
                for metric in self.config.evaluation.metrics:
                    try:
                        evaluation = await self.llm_client.evaluate_metric(
                            metric_name=metric,
                            claim=state.claim,
                            dataset_summary=state.dataset_summary,
                            task_description=state.task_description,
                            data_analysis_results=analysis_summary
                        )
                        metric_results[metric] = evaluation
                    except Exception as e:
                        logger.error(f"Error evaluating {metric}: {str(e)}")
                        metric_results[metric] = MetricEvaluation(
                            score=1,
                            rationale=f"Error: {str(e)}",
                            metadata={"error": str(e)}
                        )
            
            # Create EvaluationScores object
            state.evaluation_scores = EvaluationScores(
                correctness=metric_results["correctness"],
                helpfulness=metric_results["helpfulness"],
                complexity=metric_results["complexity"],
                coherence=metric_results["coherence"],
                verbosity=metric_results["verbosity"]
            )
            
            eval_time = time.time() - start_time
            logger.info(f"Metrics evaluation completed in {eval_time:.2f}s")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in evaluate metrics node: {str(e)}")
            state.errors.append(f"Metrics evaluation error: {str(e)}")
            return state
    
    async def _output_node(self, state: EvaluationState) -> EvaluationState:
        """Final node that formats the output"""
        logger.info("Formatting final output")
        
        try:
            if not state.evaluation_scores:
                raise ValueError("No evaluation scores available")
            
            # Create final output
            final_output = {
                "claim": state.claim,
                **state.evaluation_scores.to_dict(),
                "data_analysis_summary": self._create_analysis_summary(state.data_analysis_results),
                "timestamp": state.timestamp.isoformat(),
                "errors": state.errors if state.errors else None
            }
            
            state.final_output = final_output
            
            logger.info(f"Evaluation completed. Average score: {state.evaluation_scores.get_average_score():.2f}")
            return state
            
        except Exception as e:
            logger.error(f"Error in output node: {str(e)}")
            state.errors.append(f"Output formatting error: {str(e)}")
            return state
    
    def _get_default_dataset_summary(self) -> str:
        """Get default dataset summary for Big Mart Sales"""
        return """
Big Mart Sales Dataset:
- 8,523 records of retail sales data
- Columns: Item_Identifier, Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP, Outlet_Identifier, Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type, Outlet_Type, Item_Outlet_Sales
- Contains item details (weight, fat content, type, MRP) and outlet information (size, location, type, establishment year)
- Target variable: Item_Outlet_Sales
- Includes various product categories like Dairy, Soft Drinks, Meat, Fruits and Vegetables, etc.
- Outlet types include Supermarket Type1/2/3 and Grocery Store
- Location types: Tier 1, 2, 3
- Establishment years range from 1985 to 2009
"""
    
    def _format_analysis_for_metrics(self, analysis_results: List[DataAnalysisResult]) -> List[Dict[str, Any]]:
        """Format analysis results for use in metric evaluation"""
        formatted = []
        for result in analysis_results:
            formatted.append({
                "analysis_type": result.analysis_type,
                "results": result.results,
                "success": result.success
            })
        return formatted
    
    def _create_analysis_summary(self, analysis_results: List[DataAnalysisResult]) -> str:
        """Create a summary of data analysis results"""
        if not analysis_results:
            return "No data analysis was performed."
        
        summary_parts = []
        for result in analysis_results:
            if result.success:
                summary_parts.append(f"✓ {result.analysis_type}: {str(result.results)[:100]}...")
            else:
                summary_parts.append(f"✗ {result.analysis_type}: Failed - {result.error_message}")
        
        return "\n".join(summary_parts)
    
    async def evaluate_claim(
        self, 
        claim: str, 
        dataset_summary: str = None, 
        task_description: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate a claim using the complete workflow
        
        Args:
            claim: The claim to evaluate
            dataset_summary: Optional dataset summary
            task_description: Optional task description
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            start_time = time.time()
            
            # Create initial state
            initial_state = EvaluationState(
                claim=claim,
                dataset_summary=dataset_summary or "",
                task_description=task_description
            )
            
            # Run the workflow
            final_state = await self.graph.ainvoke(initial_state)
            
            # Calculate total execution time
            total_time = time.time() - start_time
            final_state.execution_time = total_time
            
            logger.info(f"Total evaluation time: {total_time:.2f}s")
            
            return final_state.final_output
            
        except Exception as e:
            logger.error(f"Error in evaluation workflow: {str(e)}")
            return {
                "claim": claim,
                "error": str(e),
                "scores": {"correctness": 1, "helpfulness": 1, "complexity": 1, "coherence": 1, "verbosity": 1},
                "explanations": {"error": f"Workflow error: {str(e)}"},
                "average_score": 1.0
            } 