"""
Debug version to test data analysis step only
"""
import asyncio
import json
from loguru import logger
from src.graph import EvaluationGraph

async def test_data_analysis_only():
    """Test only the data analysis step to verify it's working"""
    
    logger.info("🔍 Testing data analysis step only...")
    
    # Initialize the evaluation graph
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
    
    try:
        # Create initial state
        from src.models import EvaluationState
        initial_state = EvaluationState(
            claim=test_claim,
            dataset_info=dataset_info,
            dataset_path=dataset_path
        )
        
        logger.info("Starting data analysis step...")
        
        # Run only the data analysis node
        state = await graph._data_analysis_node(initial_state)
        
        # Check results
        logger.info("✅ Data analysis step completed!")
        
        print("\n" + "="*60)
        print("DATA ANALYSIS RESULTS")
        print("="*60)
        
        print(f"Claim: {state.claim}")
        print(f"Success: {state.data_analysis_results.success}")
        
        if state.data_analysis_results:
            print(f"\nSTDOUT ({len(state.data_analysis_results.stdout)} lines):")
            if isinstance(state.data_analysis_results.stdout, list):
                for i, line in enumerate(state.data_analysis_results.stdout[:10]):  # Show first 10 lines
                    print(f"  {i+1}: {line}")
                if len(state.data_analysis_results.stdout) > 10:
                    print(f"  ... and {len(state.data_analysis_results.stdout) - 10} more lines")
            else:
                print(f"  {state.data_analysis_results.stdout[:500]}...")
            
            print(f"\nSTDERR ({len(state.data_analysis_results.stderr)} lines):")
            if state.data_analysis_results.stderr:
                if isinstance(state.data_analysis_results.stderr, list):
                    for line in state.data_analysis_results.stderr:
                        print(f"  ERROR: {line}")
                else:
                    print(f"  ERROR: {state.data_analysis_results.stderr}")
            else:
                print("  No errors")
        
        if state.errors:
            print(f"\nState Errors:")
            for error in state.errors:
                print(f"  - {error}")
        
        # Save debug results
        debug_result = {
            "claim": state.claim,
            "success": state.data_analysis_results.success if state.data_analysis_results else False,
            "stdout": state.data_analysis_results.stdout if state.data_analysis_results else None,
            "stderr": state.data_analysis_results.stderr if state.data_analysis_results else None,
            "errors": state.errors
        }
        
        with open('debug_data_analysis.json', 'w') as f:
            json.dump(debug_result, f, indent=2, default=str)
        
        logger.info("Debug results saved to debug_data_analysis.json")
        
        return state.data_analysis_results.success if state.data_analysis_results else False
        
    except Exception as e:
        logger.error(f"❌ Data analysis test failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="{time:HH:mm:ss} | {level} | {message}"
    )
    
    # Run the debug test
    asyncio.run(test_data_analysis_only()) 