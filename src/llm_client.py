"""
Ollama LLM client - just handles connection and basic text generation
"""
import asyncio
from typing import List, Union
from langchain_ollama import OllamaLLM
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from loguru import logger

class OllamaClient:
    """Basic client for interacting with Ollama LLM"""
    
    def __init__(self, model_name: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
        # Initialize Ollama client
        self.llm = OllamaLLM(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.1,
            timeout=60  # Reduced timeout for better performance
        )
        
        logger.info(f"Initialized Ollama client with model: {self.model_name}")
    
    async def ainvoke(self, messages: List[Union[BaseMessage, dict]]) -> str:
        """Async method to generate response from Ollama"""
        try:
            logger.info(f"Calling LLM with {len(messages)} messages...")
            
            # Convert dict messages to BaseMessage objects if needed
            processed_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    if msg.get("role") == "system":
                        processed_messages.append(SystemMessage(content=msg["content"]))
                    elif msg.get("role") == "user":
                        processed_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        # Default to human message
                        processed_messages.append(HumanMessage(content=msg.get("content", "")))
                else:
                    processed_messages.append(msg)
            
            # Use the modern async interface
            response = await self.llm.ainvoke(processed_messages)
            
            logger.info("LLM response received successfully")
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"Error getting LLM response: {type(e).__name__}: {str(e)}")
            logger.error(f"Exception details: {e}")
            raise
    
    async def generate_response(self, messages: List[Union[BaseMessage, dict]]) -> str:
        """Generate response from Ollama (legacy method)"""
        return await self.ainvoke(messages)
    
    def invoke(self, messages: List[Union[BaseMessage, dict]]) -> str:
        """Synchronous method for compatibility"""
        try:
            logger.info(f"Calling LLM synchronously with {len(messages)} messages...")
            
            # Convert dict messages to BaseMessage objects if needed
            processed_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    if msg.get("role") == "system":
                        processed_messages.append(SystemMessage(content=msg["content"]))
                    elif msg.get("role") == "user":
                        processed_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        # Default to human message
                        processed_messages.append(HumanMessage(content=msg.get("content", "")))
                else:
                    processed_messages.append(msg)
            
            # Use the modern sync interface
            response = self.llm.invoke(processed_messages)
            
            logger.info("LLM response received successfully")
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"Error getting LLM response: {type(e).__name__}: {str(e)}")
            logger.error(f"Exception details: {e}")
            raise 