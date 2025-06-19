"""
Data models for the LLM Insight Evaluation Agent
"""
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime

class DataAnalysisResult(BaseModel):
    """Results from data analysis"""
    stdout: Union[str, List[str]]
    stderr: Union[str, List[str]]
    success: bool
    execution_time: float = 0.0

class MetricEvaluation(BaseModel):
    """Individual metric evaluation result"""
    score: int
    rationale: str
    metric_name: str

class EvaluationState(BaseModel):
    """State that flows through the LangGraph workflow"""
    # Input data
    claim: str
    dataset_info: str
    dataset_path: str
    
    # Analysis results
    data_analysis_results: Optional[DataAnalysisResult] = None
    
    # Evaluation results
    metric_scores: Dict[str, MetricEvaluation] = {}
    
    # Final output
    final_output: Optional[Dict[str, Any]] = None
    
    # Metadata
    timestamp: datetime = datetime.now()
    errors: List[str] = []

class TestClaim(BaseModel):
    """Test claim with known expected scores"""
    claim: str = Field(description="The test claim")
    expected_scores: Dict[str, int] = Field(description="Expected scores for each metric")
    expected_rationale: Optional[str] = Field(default=None, description="Expected reasoning")
    category: str = Field(description="Category of the claim (correct, partial, incorrect)")
    difficulty: str = Field(default="medium", description="Difficulty level of the claim")

class EvaluationResult(BaseModel):
    """Complete evaluation result"""
    claim: str
    scores: Dict[str, int]
    explanations: Dict[str, str]
    average_score: float
    data_analysis_summary: Optional[str] = None
    execution_time: float
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable format"""
        return {
            "claim": self.claim,
            "scores": self.scores,
            "explanations": self.explanations,
            "average_score": round(self.average_score, 2),
            "data_analysis_summary": self.data_analysis_summary,
            "execution_time": round(self.execution_time, 2),
            "timestamp": self.timestamp.isoformat()
        } 