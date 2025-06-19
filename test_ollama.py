"""
Quick test to verify Ollama connection
"""
import asyncio
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

async def test_ollama():
    """Test basic Ollama connection and response"""
    
    # Initialize Ollama client
    llm = OllamaLLM(
        model="llama3.2:3b",
        base_url="http://localhost:11434",
        temperature=0.1,
        timeout=60
    )
    
    logger.info("Testing Ollama connection...")
    
    # Simple test message
    messages = [
        SystemMessage(content="You are a helpful assistant. Respond briefly."),
        HumanMessage(content="Say 'Hello, Ollama is working!'")
    ]
    
    try:
        # Test async call
        if hasattr(llm, 'ainvoke'):
            response = await llm.ainvoke(messages)
        else:
            # Fallback to sync call
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, llm.invoke, messages)
        
        result = response.content if hasattr(response, 'content') else str(response)
        
        logger.info(f"Ollama response: {result}")
        logger.info("✅ Ollama connection successful!")
        
    except Exception as e:
        logger.error(f"❌ Ollama connection failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_ollama()) 