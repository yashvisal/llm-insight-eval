"""
Test data generator for validating the evaluation system
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .models import TestClaim

class TestClaimGenerator:
    """Generator for test claims with known expected scores"""
    
    def __init__(self, dataset_path: str = "data/train.csv"):
        self.data = pd.read_csv(dataset_path)
        self.known_patterns = self._analyze_dataset()
    
    def _analyze_dataset(self) -> Dict[str, Any]:
        """Analyze the dataset to understand patterns for generating test claims"""
        patterns = {}
        
        # Basic statistics
        patterns["total_records"] = len(self.data)
        patterns["columns"] = list(self.data.columns)
        
        # Sales statistics
        patterns["sales_stats"] = {
            "mean": self.data["Item_Outlet_Sales"].mean(),
            "median": self.data["Item_Outlet_Sales"].median(),
            "std": self.data["Item_Outlet_Sales"].std(),
            "min": self.data["Item_Outlet_Sales"].min(),
            "max": self.data["Item_Outlet_Sales"].max()
        }
        
        # MRP statistics
        patterns["mrp_stats"] = {
            "mean": self.data["Item_MRP"].mean(),
            "median": self.data["Item_MRP"].median(),
            "std": self.data["Item_MRP"].std(),
            "min": self.data["Item_MRP"].min(),
            "max": self.data["Item_MRP"].max()
        }
        
        # Item type distribution
        patterns["item_types"] = self.data["Item_Type"].value_counts().to_dict()
        
        # Outlet type distribution
        patterns["outlet_types"] = self.data["Outlet_Type"].value_counts().to_dict()
        
        # Fat content distribution
        patterns["fat_content"] = self.data["Item_Fat_Content"].value_counts().to_dict()
        
        # Correlations
        patterns["correlations"] = {
            "mrp_sales": self.data["Item_MRP"].corr(self.data["Item_Outlet_Sales"]),
            "visibility_sales": self.data["Item_Visibility"].corr(self.data["Item_Outlet_Sales"])
        }
        
        return patterns
    
    def generate_test_claims(self, num_claims: int = 50) -> List[TestClaim]:
        """Generate test claims with known expected scores"""
        claims = []
        
        # Generate correct claims (score 4-5)
        correct_claims = self._generate_correct_claims(num_claims // 3)
        claims.extend(correct_claims)
        
        # Generate partially correct claims (score 2-3)
        partial_claims = self._generate_partial_claims(num_claims // 3)
        claims.extend(partial_claims)
        
        # Generate incorrect claims (score 1-2)
        incorrect_claims = self._generate_incorrect_claims(num_claims // 3)
        claims.extend(incorrect_claims)
        
        # Shuffle the claims
        np.random.shuffle(claims)
        
        return claims
    
    def _generate_correct_claims(self, num_claims: int) -> List[TestClaim]:
        """Generate claims that are factually correct"""
        claims = []
        
        # Based on actual correlations
        if self.known_patterns["correlations"]["mrp_sales"] > 0.3:
            claims.append(TestClaim(
                claim="Items with higher MRP tend to have higher sales.",
                expected_scores={"correctness": 5, "helpfulness": 4, "complexity": 3, "coherence": 5, "verbosity": 4},
                category="correct",
                difficulty="easy"
            ))
        
        # Based on item type analysis
        top_item_type = max(self.known_patterns["item_types"], key=self.known_patterns["item_types"].get)
        claims.append(TestClaim(
            claim=f"{top_item_type} items have the highest sales volume in the dataset.",
            expected_scores={"correctness": 5, "helpfulness": 4, "complexity": 3, "coherence": 5, "verbosity": 4},
            category="correct",
            difficulty="easy"
        ))
        
        # Based on outlet type analysis
        top_outlet_type = max(self.known_patterns["outlet_types"], key=self.known_patterns["outlet_types"].get)
        claims.append(TestClaim(
            claim=f"{top_outlet_type} outlets contribute significantly to total sales.",
            expected_scores={"correctness": 5, "helpfulness": 4, "complexity": 3, "coherence": 5, "verbosity": 4},
            category="correct",
            difficulty="easy"
        ))
        
        # More complex correct claims
        claims.append(TestClaim(
            claim="There is a positive correlation between item MRP and sales, suggesting premium pricing strategies may be effective.",
            expected_scores={"correctness": 5, "helpfulness": 5, "complexity": 4, "coherence": 5, "verbosity": 4},
            category="correct",
            difficulty="medium"
        ))
        
        claims.append(TestClaim(
            claim="Sales performance varies significantly across different outlet types, with supermarkets showing higher average sales than grocery stores.",
            expected_scores={"correctness": 5, "helpfulness": 5, "complexity": 4, "coherence": 5, "verbosity": 4},
            category="correct",
            difficulty="medium"
        ))
        
        return claims[:num_claims]
    
    def _generate_partial_claims(self, num_claims: int) -> List[TestClaim]:
        """Generate claims that are partially correct or have caveats"""
        claims = []
        
        # Partially correct - oversimplified
        claims.append(TestClaim(
            claim="All expensive items sell better than cheap items.",
            expected_scores={"correctness": 2, "helpfulness": 2, "complexity": 1, "coherence": 4, "verbosity": 3},
            category="partial",
            difficulty="easy"
        ))
        
        # Partially correct - missing context
        claims.append(TestClaim(
            claim="Dairy products have high sales.",
            expected_scores={"correctness": 3, "helpfulness": 2, "complexity": 1, "coherence": 4, "verbosity": 2},
            category="partial",
            difficulty="easy"
        ))
        
        # Partially correct - vague
        claims.append(TestClaim(
            claim="Some items sell better than others.",
            expected_scores={"correctness": 3, "helpfulness": 1, "complexity": 1, "coherence": 3, "verbosity": 2},
            category="partial",
            difficulty="easy"
        ))
        
        # Partially correct - correlation vs causation
        claims.append(TestClaim(
            claim="Higher MRP causes higher sales, so we should increase all prices.",
            expected_scores={"correctness": 2, "helpfulness": 2, "complexity": 3, "coherence": 4, "verbosity": 4},
            category="partial",
            difficulty="medium"
        ))
        
        return claims[:num_claims]
    
    def _generate_incorrect_claims(self, num_claims: int) -> List[TestClaim]:
        """Generate claims that are factually incorrect"""
        claims = []
        
        # Factually incorrect
        claims.append(TestClaim(
            claim="Items with lower MRP have significantly higher sales than expensive items.",
            expected_scores={"correctness": 1, "helpfulness": 1, "complexity": 2, "coherence": 4, "verbosity": 4},
            category="incorrect",
            difficulty="easy"
        ))
        
        # Contradicts data
        claims.append(TestClaim(
            claim="There is no relationship between item price and sales performance.",
            expected_scores={"correctness": 1, "helpfulness": 1, "complexity": 2, "coherence": 4, "verbosity": 3},
            category="incorrect",
            difficulty="easy"
        ))
        
        # False generalization
        claims.append(TestClaim(
            claim="All grocery stores outperform all supermarkets in sales.",
            expected_scores={"correctness": 1, "helpfulness": 1, "complexity": 2, "coherence": 4, "verbosity": 3},
            category="incorrect",
            difficulty="easy"
        ))
        
        # Impossible claim
        claims.append(TestClaim(
            claim="Items with negative prices have the highest sales.",
            expected_scores={"correctness": 1, "helpfulness": 1, "complexity": 1, "coherence": 3, "verbosity": 3},
            category="incorrect",
            difficulty="easy"
        ))
        
        # Contradicts basic statistics
        claims.append(TestClaim(
            claim="The average item sells for over $1000, which is clearly too expensive for this market.",
            expected_scores={"correctness": 1, "helpfulness": 2, "complexity": 2, "coherence": 4, "verbosity": 4},
            category="incorrect",
            difficulty="medium"
        ))
        
        return claims[:num_claims]
    
    def generate_validation_dataset(self, num_claims: int = 50) -> Dict[str, Any]:
        """Generate a complete validation dataset"""
        claims = self.generate_test_claims(num_claims)
        
        # Convert to dictionary format
        validation_data = {
            "dataset_info": {
                "total_claims": len(claims),
                "categories": {
                    "correct": len([c for c in claims if c.category == "correct"]),
                    "partial": len([c for c in claims if c.category == "partial"]),
                    "incorrect": len([c for c in claims if c.category == "incorrect"])
                },
                "difficulty_levels": {
                    "easy": len([c for c in claims if c.difficulty == "easy"]),
                    "medium": len([c for c in claims if c.difficulty == "medium"]),
                    "hard": len([c for c in claims if c.difficulty == "hard"])
                }
            },
            "claims": [
                {
                    "claim": c.claim,
                    "expected_scores": c.expected_scores,
                    "category": c.category,
                    "difficulty": c.difficulty,
                    "expected_rationale": c.expected_rationale
                }
                for c in claims
            ]
        }
        
        return validation_data
    
    def save_validation_dataset(self, filename: str = "validation_dataset.json"):
        """Save the validation dataset to a file"""
        import json
        
        validation_data = self.generate_validation_dataset()
        
        with open(filename, 'w') as f:
            json.dump(validation_data, f, indent=2)
        
        print(f"Validation dataset saved to {filename}")
        print(f"Generated {len(validation_data['claims'])} test claims")
        print(f"Categories: {validation_data['dataset_info']['categories']}") 