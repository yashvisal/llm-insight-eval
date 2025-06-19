"""
Ollama LLM client - just handles connection and basic text generation
"""
import asyncio
from typing import List
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

class OllamaClient:
    """Basic client for interacting with Ollama LLM"""
    
    def __init__(self, model_name: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
        # Initialize Ollama client
        self.llm = Ollama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.1,
            timeout=60
        )
        
        logger.info(f"Initialized Ollama client with model: {self.model_name}")
    
    async def ainvoke(self, messages: List) -> str:
        """Async method to generate response from Ollama"""
        try:
            # Convert to async if needed
            if hasattr(self.llm, 'ainvoke'):
                response = await self.llm.ainvoke(messages)
            else:
                # Fallback to sync call
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, self.llm.invoke, messages)
            
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            raise
    
    async def generate_response(self, messages: List) -> str:
        """Generate response from Ollama (legacy method)"""
        return await self.ainvoke(messages) 