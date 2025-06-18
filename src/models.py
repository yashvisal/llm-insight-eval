"""
Data models for the LLM Insight Evaluation Agent
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class DataAnalysisResult(BaseModel):
    """Results from data analysis operations"""
    analysis_type: str = Field(description="Type of analysis performed")
    results: Dict[str, Any] = Field(description="Analysis results")
    code_executed: Optional[str] = Field(default=None, description="Code that was executed")
    execution_time: float = Field(description="Time taken for analysis in seconds")
    success: bool = Field(description="Whether the analysis was successful")
    error_message: Optional[str] = Field(default=None, description="Error message if analysis failed")

class MetricEvaluation(BaseModel):
    """Individual metric evaluation result"""
    score: int = Field(ge=1, le=5, description="Evaluation score from 1-5")
    rationale: str = Field(description="Explanation for the score")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence in the evaluation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class EvaluationScores(BaseModel):
    """All evaluation scores for a claim"""
    correctness: MetricEvaluation
    helpfulness: MetricEvaluation
    complexity: MetricEvaluation
    coherence: MetricEvaluation
    verbosity: MetricEvaluation
    
    def get_average_score(self) -> float:
        """Calculate the average score across all metrics"""
        scores = [
            self.correctness.score,
            self.helpfulness.score,
            self.complexity.score,
            self.coherence.score,
            self.verbosity.score
        ]
        return sum(scores) / len(scores)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for output"""
        return {
            "scores": {
                "correctness": self.correctness.score,
                "helpfulness": self.helpfulness.score,
                "complexity": self.complexity.score,
                "coherence": self.coherence.score,
                "verbosity": self.verbosity.score
            },
            "explanations": {
                "correctness": self.correctness.rationale,
                "helpfulness": self.helpfulness.rationale,
                "complexity": self.complexity.rationale,
                "coherence": self.coherence.rationale,
                "verbosity": self.verbosity.rationale
            },
            "average_score": self.get_average_score()
        }

class EvaluationState(BaseModel):
    """State management for the evaluation workflow"""
    # Input data
    claim: str = Field(description="The claim/insight to evaluate")
    dataset_summary: str = Field(description="Summary of the dataset")
    task_description: Optional[str] = Field(default=None, description="Description of the task context")
    
    # Analysis results
    data_analysis_results: List[DataAnalysisResult] = Field(
        default_factory=list, 
        description="Results from data analysis operations"
    )
    
    # Evaluation results
    evaluation_scores: Optional[EvaluationScores] = Field(
        default=None, 
        description="Evaluation scores for all metrics"
    )
    
    # Final output
    final_output: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Final formatted output"
    )
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now, description="When the evaluation was performed")
    execution_time: Optional[float] = Field(default=None, description="Total execution time")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")

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