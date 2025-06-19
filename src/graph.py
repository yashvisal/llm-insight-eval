"""
LangGraph workflow for LLM Insight Evaluation Agent
"""
import asyncio
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from loguru import logger

from .models import EvaluationState

class EvaluationGraph:
    """Main evaluation workflow using LangGraph"""
    
    def __init__(self):
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
        # TODO: Implement E2B integration
        logger.info("Data analysis completed (placeholder)")
        return state
    
    async def _evaluate_metrics_node(self, state: EvaluationState) -> EvaluationState:
        """Node that evaluates all metrics in parallel"""
        logger.info("Evaluating metrics")
        # TODO: Implement metric evaluation
        logger.info("Metrics evaluation completed (placeholder)")
        return state
    
    async def _output_node(self, state: EvaluationState) -> EvaluationState:
        """Final node that formats the output"""
        logger.info("Formatting final output")
        
        # Create final output
        final_output = {
            "claim": state.claim,
            "dataset_info": state.dataset_info,
            "status": "completed",
            "timestamp": state.timestamp.isoformat(),
            "data_analysis": "placeholder",
            "metrics": "placeholder",
            "errors": state.errors if state.errors else []
        }
        
        # Update state with final output
        state.final_output = final_output
        logger.info("Output formatting completed (placeholder)")
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