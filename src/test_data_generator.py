import pandas as pd

class TestClaimGenerator:
    def __init__(self, dataset_path: str):
        self.data = pd.read_csv(dataset_path)
        self.known_patterns = self._analyze_dataset()
    
    def generate_test_claims(self, num_claims: int = 50):
        """Generate claims with known expected scores"""
        claims = []
        
        # Generate correct claims (score 4-5)
        claims.extend(self._generate_correct_claims(num_claims // 3))
        
        # Generate partially correct claims (score 2-3)
        claims.extend(self._generate_partial_claims(num_claims // 3))
        
        # Generate incorrect claims (score 1-2)
        claims.extend(self._generate_incorrect_claims(num_claims // 3))
        
        return claims 