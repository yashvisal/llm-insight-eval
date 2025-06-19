"""
LangGraph workflow for LLM Insight Evaluation Agent
"""
import asyncio
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from loguru import logger
from e2b_code_interpreter import Sandbox
from langchain_core.messages import HumanMessage, SystemMessage

from .models import EvaluationState
from .llm_client import OllamaClient
from .code_generator import CodeGenerator
from .metric_evaluator import MetricEvaluator

class EvaluationGraph:
    """Main evaluation workflow using LangGraph"""
    
    def __init__(self):
        # Initialize components
        self.llm_client = OllamaClient()
        self.code_generator = CodeGenerator(self.llm_client)
        self.metric_evaluator = MetricEvaluator(self.llm_client)
        
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
        logger.info(f"Evaluating claim: {state.claim[:100]}...")
        return state
    
    async def _data_analysis_node(self, state: EvaluationState) -> EvaluationState:
        """Node that performs data analysis relevant to the claim using E2B"""
        logger.info("Performing data analysis with E2B")
        
        try:
            # Generate analysis code based on the claim
            analysis_prompt = f"""
            You are a data analyst. Write clear, efficient Python code for data analysis.
            
            Analyze the following claim about the Big Mart Sales dataset:
            Claim: {state.claim}
            
            Dataset info: {state.dataset_info}
            
            Please write Python code to:
            1. Load and explore the dataset (train.csv)
            2. Perform relevant statistical analysis to evaluate this claim
            3. Generate visualizations if helpful
            4. Return key findings and statistics
            
            Focus on data that directly relates to validating or refuting the claim.
            Only respond with the code to be executed and nothing else. Strip backticks in code blocks.
            """
            
            # Get analysis code from LLM
            messages = [
                SystemMessage(content="You are a helpful assistant that can execute python code in a Jupyter notebook. Only respond with the code to be executed and nothing else. Strip backticks in code blocks."),
                HumanMessage(content=analysis_prompt)
            ]
            
            code_response = await self.llm_client.ainvoke(messages)
            analysis_code = code_response
            
            # Execute code in E2B sandbox
            logger.info("Executing analysis code in E2B sandbox")
            
            with Sandbox() as sandbox:
                # Upload the dataset file
                with open(state.dataset_path, 'r') as f:
                    dataset_content = f.read()
                
                # Upload files in E2B
                sandbox.files.write("train.csv", dataset_content)
                
                # Execute the analysis code
                execution = sandbox.run_code(analysis_code)
                
                # Get results
                stdout = execution.logs.stdout
                stderr = execution.logs.stderr
                success = not bool(stderr)
                
            # Create analysis result
            from .models import DataAnalysisResult
            analysis_result = DataAnalysisResult(
                stdout=stdout,
                stderr=stderr,
                success=success,
                execution_time=0.0
            )
            
            # Update state with analysis results
            state.data_analysis_results = analysis_result
            
            logger.info("Data analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in data analysis: {str(e)}")
            state.errors.append(f"Data analysis error: {str(e)}")
            
            # Create error result
            from .models import DataAnalysisResult
            state.data_analysis_results = DataAnalysisResult(
                stdout="",
                stderr=str(e),
                success=False,
                execution_time=0.0
            )
        
        return state
    
    async def _evaluate_metrics_node(self, state: EvaluationState) -> EvaluationState:
        """Node that evaluates all metrics in parallel"""
        logger.info("Evaluating metrics")
        
        # Check if data analysis was successful before proceeding
        if not state.data_analysis_results or not state.data_analysis_results.success:
            logger.warning("Data analysis failed, skipping metric evaluation")
            # Set default low scores for all metrics
            metrics = ["correctness", "helpfulness", "complexity", "coherence", "verbosity"]
            for metric in metrics:
                from .models import MetricEvaluation
                state.metric_scores[metric] = MetricEvaluation(
                    score=1,  # Low score due to failed analysis
                    rationale=f"Evaluation based on failed data analysis: {state.data_analysis_results.stderr if state.data_analysis_results else 'No analysis performed'}",
                    metric_name=metric
                )
            return state
        
        try:
            # Prepare context for metric evaluation
            analysis_summary = f"Data Analysis Results:\n{state.data_analysis_results.stdout}"
            
            # Define metrics to evaluate
            metrics = ["correctness", "helpfulness", "complexity", "coherence", "verbosity"]
            
            # Evaluate each metric
            for metric in metrics:
                metric_result = await self.metric_evaluator.evaluate_metric(
                    claim=state.claim,
                    dataset_info=state.dataset_info,
                    analysis_results=analysis_summary,
                    metric_name=metric
                )
                
                state.metric_scores[metric] = metric_result
            
            logger.info("Metrics evaluation completed")
            
        except Exception as e:
            logger.error(f"Error in metrics evaluation: {str(e)}")
            state.errors.append(f"Metrics evaluation error: {str(e)}")
        
        return state
    
    async def _output_node(self, state: EvaluationState) -> EvaluationState:
        """Final node that formats the output"""
        logger.info("Formatting final output")
        
        # Calculate average score
        scores = {}
        explanations = {}
        total_score = 0
        metric_count = 0
        
        for metric_name, metric_result in state.metric_scores.items():
            scores[metric_name] = metric_result.score
            explanations[metric_name] = metric_result.rationale
            total_score += metric_result.score
            metric_count += 1
        
        average_score = total_score / metric_count if metric_count > 0 else 0
        
        # Create final output
        final_output = {
            "claim": state.claim,
            "dataset_info": state.dataset_info,
            "status": "completed",
            "timestamp": state.timestamp.isoformat(),
            "scores": scores,
            "explanations": explanations,
            "average_score": round(average_score, 2),
            "data_analysis_summary": '\n'.join(state.data_analysis_results.stdout) if state.data_analysis_results and state.data_analysis_results.stdout else None,
            "errors": state.errors if state.errors else []
        }
        
        # Update state with final output
        state.final_output = final_output
        logger.info("Output formatting completed")
        return state
    
    async def evaluate_claim(
        self, 
        claim: str, 
        dataset_info: str,
        dataset_path: str
    ) -> Dict[str, Any]:
        """Main entry point for claim evaluation"""
        # Create initial state
        initial_state = EvaluationState(
            claim=claim,
            dataset_info=dataset_info,
            dataset_path=dataset_path
        )
        
        # Run the workflow
        final_state = await self.graph.ainvoke(initial_state)
        
        # Convert back to Pydantic model if needed
        if hasattr(final_state, 'final_output'):
            return final_state.final_output
        else:
            # Handle case where state is converted to dict
            return final_state.get('final_output', {"status": "completed"}) 