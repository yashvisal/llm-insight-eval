"""
Configuration management for the LLM Insight Evaluation Agent
"""
import os
from pathlib import Path
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class LLMConfig(BaseModel):
    """Configuration for LLM settings"""
    model_name: str = Field(default="llama3.2", description="Ollama model name")
    base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    temperature: float = Field(default=0.1, description="LLM temperature for deterministic outputs")
    max_tokens: int = Field(default=2048, description="Maximum tokens for LLM responses")
    timeout: int = Field(default=120, description="Timeout for LLM requests in seconds")

class DatasetConfig(BaseModel):
    """Configuration for dataset settings"""
    data_path: str = Field(default="data/train.csv", description="Path to the dataset")
    dataset_name: str = Field(default="Big Mart Sales", description="Name of the dataset")
    dataset_description: str = Field(
        default="Retail sales data from Big Mart stores with item and outlet information",
        description="Brief description of the dataset"
    )

class EvaluationConfig(BaseModel):
    """Configuration for evaluation settings"""
    metrics: list[str] = Field(
        default=["correctness", "helpfulness", "complexity", "coherence", "verbosity"],
        description="List of evaluation metrics to use"
    )
    score_range: tuple[int, int] = Field(default=(1, 5), description="Score range for metrics")
    parallel_evaluation: bool = Field(default=True, description="Whether to run evaluations in parallel")
    max_analysis_time: int = Field(default=300, description="Maximum time for data analysis in seconds")

class E2BConfig(BaseModel):
    """E2B configuration"""
    api_key: str = Field(default="", description="E2B API key")
    enabled: bool = Field(default=True, description="Whether to use E2B for data analysis")
    timeout: int = Field(default=300, description="E2B sandbox timeout in seconds")
    auto_install_packages: List[str] = Field(
        default=["pandas", "numpy", "matplotlib", "seaborn", "scipy", "scikit-learn"],
        description="Python packages to auto-install in E2B sandbox"
    )

class Config(BaseModel):
    """Main configuration class"""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging_level: str = Field(default="INFO", description="Logging level")
    output_dir: str = Field(default="outputs", description="Directory for output files")
    e2b: E2BConfig = Field(default_factory=E2BConfig, description="E2B configuration")

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get configuration from environment variables and defaults"""
    load_dotenv()
    
    return Config(
        # ... existing config ...
        e2b=E2BConfig(
            api_key=os.getenv("E2B_API_KEY", ""),
            enabled=os.getenv("E2B_ENABLED", "true").lower() == "true",
            timeout=int(os.getenv("E2B_TIMEOUT", "300")),
            auto_install_packages=[
                "pandas", "numpy", "matplotlib", "seaborn", 
                "scipy", "scikit-learn", "plotly"
            ]
        )
    )

def update_config(**kwargs) -> None:
    """Update configuration with new values"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}") 