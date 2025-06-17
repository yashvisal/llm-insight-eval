from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from pydantic import BaseModel
import json

class EvaluationState(BaseModel):
    claim: str
    dataset_summary: str
    data_analysis_results: Dict[str, Any] = {}
    evaluation_scores: Dict[str, Dict[str, Any]] = {}
    final_output: Dict[str, Any] = {}

def create_evaluation_graph():
    # Define the workflow
    workflow = StateGraph(EvaluationState)
    
    # Add nodes
    workflow.add_node("start", start_node)
    workflow.add_node("data_analysis", data_analysis_node)
    workflow.add_node("evaluate_metrics", evaluate_metrics_node)
    workflow.add_node("output", output_node)
    
    # Define edges
    workflow.set_entry_point("start")
    workflow.add_edge("start", "data_analysis")
    workflow.add_edge("data_analysis", "evaluate_metrics")
    workflow.add_edge("evaluate_metrics", "output")
    workflow.add_edge("output", END)
    
    return workflow.compile() 