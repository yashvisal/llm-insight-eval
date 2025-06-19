"""
Main entry point for the LLM Insight Evaluation Agent
"""
import asyncio
import json
import argparse
from pathlib import Path
from loguru import logger
from src.graph import EvaluationGraph
from src.test_data_generator import TestClaimGenerator
from src.config import get_config
import os

def setup_logging():
    """Setup logging configuration"""
    logger.remove()  # Remove default handler
    logger.add(
        "logs/evaluation.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="{time:HH:mm:ss} | {level} | {message}"
    )

async def evaluate_single_claim(claim: str, dataset_summary: str = None, task_description: str = None):
    """Evaluate a single claim"""
    logger.info("Initializing evaluation system...")
    
    # Create evaluation graph
    graph = EvaluationGraph()
    
    # Run evaluation
    logger.info(f"Evaluating claim: {claim}")
    result = await graph.evaluate_claim(claim, dataset_summary, task_description)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Claim: {result['claim']}")
    print(f"Average Score: {result['average_score']:.2f}/5.0")
    print("\nIndividual Scores:")
    for metric, score in result['scores'].items():
        print(f"  {metric.capitalize()}: {score}/5")
    print("\nExplanations:")
    for metric, explanation in result['explanations'].items():
        print(f"  {metric.capitalize()}: {explanation}")
    
    if result.get('data_analysis_summary'):
        print(f"\nData Analysis: {result['data_analysis_summary']}")
    
    if result.get('errors'):
        print(f"\nErrors: {result['errors']}")
    
    return result

async def run_validation_test(num_claims: int = 10):
    """Run validation test with generated test claims"""
    logger.info("Running validation test...")
    
    # Generate test claims
    generator = TestClaimGenerator()
    test_claims = generator.generate_test_claims(num_claims)
    
    # Create evaluation graph
    graph = EvaluationGraph()
    
    results = []
    correct_predictions = 0
    
    for i, test_claim in enumerate(test_claims):
        logger.info(f"Testing claim {i+1}/{len(test_claims)}: {test_claim.claim[:50]}...")
        
        # Run evaluation
        result = await graph.evaluate_claim(test_claim.claim)
        
        # Compare with expected scores
        actual_score = result['scores']['correctness']
        expected_score = test_claim.expected_scores['correctness']
        
        # Consider it correct if within 1 point
        if abs(actual_score - expected_score) <= 1:
            correct_predictions += 1
        
        results.append({
            'claim': test_claim.claim,
            'expected': test_claim.expected_scores,
            'actual': result['scores'],
            'category': test_claim.category,
            'correct_prediction': abs(actual_score - expected_score) <= 1
        })
    
    # Print validation results
    accuracy = correct_predictions / len(test_claims) * 100
    print(f"\nValidation Results:")
    print(f"Accuracy: {accuracy:.1f}% ({correct_predictions}/{len(test_claims)})")
    
    # Save detailed results
    validation_results = {
        'accuracy': accuracy,
        'total_tests': len(test_claims),
        'correct_predictions': correct_predictions,
        'results': results
    }
    
    with open('validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    logger.info(f"Validation results saved to validation_results.json")
    return validation_results

def generate_test_dataset():
    """Generate and save test dataset"""
    logger.info("Generating test dataset...")
    
    generator = TestClaimGenerator()
    generator.save_validation_dataset("test_dataset.json")
    
    logger.info("Test dataset generated successfully!")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="LLM Insight Evaluation Agent")
    parser.add_argument("--claim", required=True, help="Claim to evaluate")
    parser.add_argument("--dataset-summary", help="Dataset summary")
    parser.add_argument("--task-description", help="Task description")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--e2b-api-key", help="E2B API key")
    parser.add_argument("--disable-e2b", action="store_true", help="Disable E2B analysis")
    parser.add_argument("--validate", action="store_true", help="Run validation test")
    parser.add_argument("--generate-test-data", action="store_true", help="Generate test dataset")
    parser.add_argument("--num-claims", type=int, default=10, help="Number of test claims for validation")
    
    args = parser.parse_args()
    
    # Set E2B API key if provided
    if args.e2b_api_key:
        os.environ["E2B_API_KEY"] = args.e2b_api_key
    
    if args.disable_e2b:
        os.environ["E2B_ENABLED"] = "false"
    
    # Setup logging
    setup_logging()
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    
    if args.generate_test_data:
        generate_test_dataset()
        return
    
    if args.validate:
        asyncio.run(run_validation_test(args.num_claims))
        return
    
    if args.claim:
        asyncio.run(evaluate_single_claim(
            args.claim, 
            args.dataset_summary, 
            args.task_description
        ))
    else:
        # Interactive mode
        print("LLM Insight Evaluation Agent")
        print("="*40)
        print("Enter a claim to evaluate (or 'quit' to exit):")
        
        while True:
            claim = input("\nClaim: ").strip()
            if claim.lower() in ['quit', 'exit', 'q']:
                break
            
            if not claim:
                continue
            
            dataset_summary = input("Dataset summary (optional): ").strip() or None
            task_description = input("Task description (optional): ").strip() or None
            
            asyncio.run(evaluate_single_claim(claim, dataset_summary, task_description))

if __name__ == "__main__":
    main() 