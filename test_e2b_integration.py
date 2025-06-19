"""
Test script for E2B integration with the LLM Insight Evaluation Agent
"""
import asyncio
import json
from loguru import logger
from src.graph import EvaluationGraph

async def test_e2b_integration():
    """Test the E2B integration with a sample claim"""
    
    # Initialize the evaluation graph
    logger.info("Initializing Evaluation Graph...")
    graph = EvaluationGraph()
    
    # Sample claim about the Big Mart Sales dataset
    test_claim = "The average item weight in the Big Mart Sales dataset is around 12.5 kg"
    
    # Dataset information
    dataset_info = """
    Big Mart Sales Dataset contains:
    - Item_Identifier: Unique identifier for each item
    - Item_Weight: Weight of the item in kg
    - Item_Fat_Content: Fat content category (Low Fat, Regular, etc.)
    - Item_Visibility: Visibility percentage of the item
    - Item_Type: Category of the item (Dairy, Soft Drinks, etc.)
    - Item_MRP: Maximum Retail Price
    - Outlet_Identifier: Unique identifier for each outlet
    - Outlet_Establishment_Year: Year when outlet was established
    - Outlet_Size: Size of the outlet (Small, Medium, High)
    - Outlet_Location_Type: Location tier (Tier 1, Tier 2, Tier 3)
    - Outlet_Type: Type of outlet (Supermarket Type1, Grocery Store, etc.)
    - Item_Outlet_Sales: Sales of the item in the outlet
    """
    
    # Dataset path
    dataset_path = "data/train.csv"
    
    logger.info(f"Testing claim: {test_claim}")
    logger.info("Starting evaluation workflow...")
    
    try:
        # Run the evaluation
        result = await graph.evaluate_claim(
            claim=test_claim,
            dataset_info=dataset_info,
            dataset_path=dataset_path
        )
        
        # Print results
        logger.info("✅ Evaluation completed successfully!")
        logger.info("\n" + "="*50)
        logger.info("EVALUATION RESULTS")
        logger.info("="*50)
        
        print(f"Claim: {result['claim']}")
        print(f"Status: {result['status']}")
        print(f"Average Score: {result.get('average_score', 'N/A')}")
        
        if 'scores' in result:
            print("\nMetric Scores:")
            for metric, score in result['scores'].items():
                print(f"  {metric.capitalize()}: {score}/10")
        
        if 'explanations' in result:
            print("\nMetric Explanations:")
            for metric, explanation in result['explanations'].items():
                print(f"\n  {metric.capitalize()}:")
                print(f"    {explanation[:200]}...")
        
        if result.get('data_analysis_summary'):
            print(f"\nData Analysis Summary:")
            print(f"  {result['data_analysis_summary'][:300]}...")
        
        if result.get('errors'):
            print(f"\nErrors encountered:")
            for error in result['errors']:
                print(f"  - {error}")
        
        # Save results to file
        with open('test_results.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info("\nResults saved to test_results.json")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Configure logging
    logger.add("test_e2b.log", rotation="1 MB")
    
    # Run the test
    asyncio.run(test_e2b_integration()) 