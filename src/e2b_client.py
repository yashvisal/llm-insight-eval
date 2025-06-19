"""
E2B client
"""
import os
from typing import Dict, Any, Optional
from e2b_code_interpreter import Sandbox
from loguru import logger

class E2BClient:
    """E2B client for data analysis"""
    
    def __init__(self):
        self.api_key = os.getenv("E2B_API_KEY")
        self.sandbox = None
    
    def is_available(self) -> bool:
        """Check if E2B is available"""
        return self.api_key is not None
    
    def upload_dataset(self, dataset_path: str) -> Optional[str]:
        """Upload dataset to E2B sandbox"""
        if not self.is_available():
            logger.warning("E2B API key not provided")
            return None
        
        try:
            with Sandbox() as sandbox:
                self.sandbox = sandbox
                with open(dataset_path, "rb") as f:
                    remote_path = sandbox.upload_file(f, "dataset.csv")
                logger.info(f"Dataset uploaded: {remote_path}")
                return remote_path
        except Exception as e:
            logger.error(f"Error uploading dataset: {str(e)}")
            return None
    
    def run_analysis(self, code: str) -> Dict[str, Any]:
        """Run Python code in E2B sandbox"""
        if not self.is_available():
            return {"error": "E2B not available"}
        
        try:
            with Sandbox() as sandbox:
                execution = sandbox.run_code(code)
                return {
                    "success": True,
                    "stdout": execution.text,
                    "stderr": execution.error if execution.error else ""
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            } 