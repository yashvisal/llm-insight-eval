"""
Configuration management for the LLM Insight Evaluation Agent
"""
import os
from pathlib import Path
from typing import Dict, Any
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

class Config(BaseModel):
    """Main configuration class"""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging_level: str = Field(default="INFO", description="Logging level")
    output_dir: str = Field(default="outputs", description="Directory for output files")

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance"""
    return config

def update_config(**kwargs) -> None:
    """Update configuration with new values"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}") 