"""
Setup script for LLM Insight Evaluation Agent
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_ollama():
    """Check if Ollama is installed and running"""
    print("🔍 Checking Ollama installation...")
    
    # Check if ollama command exists
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama command failed")
            return False
    except FileNotFoundError:
        print("❌ Ollama not found. Please install Ollama from https://ollama.ai/")
        return False

def setup_ollama_model(model_name="llama3.2"):
    """Setup the default Ollama model"""
    print(f"🔍 Checking for model: {model_name}")
    
    try:
        # Check if model exists
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model_name in result.stdout:
            print(f"✅ Model {model_name} already installed")
            return True
        else:
            print(f"📥 Installing model: {model_name}")
            return run_command(f"ollama pull {model_name}", f"Downloading {model_name}")
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ["logs", "outputs", "data"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main setup function"""
    print("🚀 Setting up LLM Insight Evaluation Agent")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check Ollama
    if not check_ollama():
        print("\n📋 To install Ollama:")
        print("1. Visit https://ollama.ai/")
        print("2. Download and install for your platform")
        print("3. Run: ollama serve")
        print("4. Run this setup script again")
        sys.exit(1)
    
    # Setup model
    if not setup_ollama_model():
        print("❌ Failed to setup Ollama model")
        sys.exit(1)
    
    # Test the system
    print("\n🧪 Testing the system...")
    test_claim = "Items with higher MRP tend to have higher sales."
    
    try:
        from src.graph import EvaluationGraph
        import asyncio
        
        async def test_evaluation():
            graph = EvaluationGraph()
            result = await graph.evaluate_claim(test_claim)
            return result
        
        result = asyncio.run(test_evaluation())
        print("✅ System test completed successfully!")
        print(f"📊 Test claim evaluated with average score: {result.get('average_score', 'N/A')}")
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        print("💡 This might be normal if Ollama is not running. Start it with: ollama serve")
    
    print("\n🎉 Setup completed!")
    print("\n📖 Next steps:")
    print("1. Start Ollama: ollama serve")
    print("2. Test the system: python main.py --claim 'Your claim here'")
    print("3. Generate test data: python main.py --generate-test-data")
    print("4. Run validation: python main.py --validate")
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main() 