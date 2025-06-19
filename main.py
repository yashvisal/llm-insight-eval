"""
Main entry point for the LLM Insight Evaluation Agent
"""
import asyncio
import argparse
from loguru import logger
from src.graph import EvaluationGraph

def setup_logging():
    """Setup logging configuration"""
    logger.remove()  # Remove default handler
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="{time:HH:mm:ss} | {level} | {message}"
    )

async def evaluate_claim(claim: str, dataset_info: str, dataset_path: str):
    """Evaluate a single claim"""
    logger.info("Initializing evaluation system...")
    
    # Create evaluation graph
    graph = EvaluationGraph()
    
    # Run evaluation
    logger.info(f"Evaluating claim: {claim}")
    result = await graph.evaluate_claim(claim, dataset_info, dataset_path)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Result: {result}")
    
    return result

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="LLM Insight Evaluation Agent")
    parser.add_argument("--claim", required=True, help="Claim to evaluate")
    parser.add_argument("--dataset-info", default="Big Mart Sales dataset", help="Dataset summary")
    parser.add_argument("--dataset-path", default="data/train.csv", help="Path to dataset")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Run evaluation
    asyncio.run(evaluate_claim(
        args.claim, 
        args.dataset_info, 
        args.dataset_path
    ))

if __name__ == "__main__":
    main() 