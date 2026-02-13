#!/usr/bin/env python3
import base64
import io, re, json
import time
import numpy as np
import uuid
import requests
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from PIL import Image
from openai import OpenAI
from dataclasses import dataclass


def image_to_base64_url(image: Union[Image.Image, np.ndarray]) -> str:
    """
    Convert PIL Image or numpy array to base64 data URL.
    
    Args:
        image: PIL Image or numpy array (H, W, C) with values 0-255
        
    Returns:
        str: Base64 encoded data URL
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    
    with io.BytesIO() as output:
        image.save(output, format="JPEG")
        encoded = base64.b64encode(output.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"


@dataclass
class LLMRequest:
    """
    Data class representing a request to the LLM server.
    
    Attributes:
        content: List of mixed content items (text strings, PIL Images, or numpy arrays)
        system_prompt: Optional system prompt for the conversation
        generation_params: Optional parameters for text generation
    """
    content: List[Union[str, Image.Image, np.ndarray]]
    system_prompt: Optional[str] = None
    generation_params: Optional[Dict[str, Any]] = None
    

class GeneralLLMServer:
    """
    General-purpose LLM server client supporting multimodal inputs and flexible post-processing.
    
    This class provides a unified interface for communicating with LLM servers that support
    OpenAI-compatible APIs, with built-in support for:
    - System prompt handling (optional)
    - Multimodal inputs (interleaved text and images)
    - Customizable post-processing
    - Retry logic and error handling
    
    Example:
        server = GeneralLLMServer(
            url="http://localhost:8080/v1",
            model_name="Qwen2.5-VL-7B-Instruct"
        )
        
        # Simple text request
        response = server.send_request(["Describe this image in detail."])
        
        # Multimodal request
        response = server.send_request([
            "Analyze this image:", 
            image,
            "What do you see?"
        ])
    """
    
    def __init__(
        self,
        url: str,
        model_name: str,
        api_key: str = "EMPTY",
        max_retries: int = 2,
        retry_delay: float = 1.0,
        default_generation_params: Optional[Dict[str, Any]] = None,
        post_processor: Optional[Callable[[str], Any]] = None,
        client_type: str = "openai"
    ):
        """
        Initialize the GeneralLLMServer.
        
        Args:
            url: Base URL for the OpenAI-compatible API server or Gemini API server
            model_name: Name of the model to use
            api_key: API key (default: "EMPTY" for local servers)
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retry attempts in seconds
            default_generation_params: Default parameters for text generation
            post_processor: Optional function to post-process responses
            client_type: Type of client ("openai" or "gemini")
        """
        self.url = url
        self.model_name = model_name
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.post_processor = post_processor
        self.client_type = client_type.lower()
        
        # Default generation parameters
        if default_generation_params is None:
            if self.client_type == "custom":
                default_generation_params = {
                    # "generationConfig": {"thinkingConfig": {"thinkingBudget": 128}}
                    "generationConfig": {"temperature": 0.0}
                }
            else:
                default_generation_params = {
                    "temperature": 0.0,
                    "max_tokens": 4096,
                    "top_p": 0.9,
                }
        self.default_generation_params = default_generation_params
        
        # Default to OpenAI client
        self.client = OpenAI(
            api_key=api_key,
            base_url=url,
        )
    
    def _prepare_system_prompt(self, system_prompt: Optional[str]) -> Optional[Dict[str, str]]:
        """
        Prepare system prompt for the message format.
        
        Args:
            system_prompt: Optional system prompt string
            
        Returns:
            Optional[Dict]: System message dict or None if no system prompt
        """
        if system_prompt and system_prompt.strip():
            return {"role": "system", "content": system_prompt.strip()}
        return None
    
    def _prepare_custom_content(self, content: List[Union[str, Image.Image, np.ndarray]]) -> List[Dict[str, Any]]:
        """
        Prepare user content for Custom API format.
        
        Args:
            content: List of mixed content items (strings, images, arrays)
            
        Returns:
            List[Dict]: Formatted content list for Custom API
        """
        formatted_content = []
        
        for item in content:
            if isinstance(item, str):
                # Text content
                formatted_content.append({
                    "type": "text",
                    "value": item
                })
            elif isinstance(item, (Image.Image, np.ndarray)):
                # Image content - Gemini uses PNG format
                formatted_content.append({
                    "type": "image_url",
                    "value": image_to_base64_url(item)
                })
            else:
                # Convert other types to string
                formatted_content.append({
                    "type": "text",
                    "value": str(item)
                })
        
        return formatted_content

    def _prepare_user_content(self, content: List[Union[str, Image.Image, np.ndarray]]) -> List[Dict[str, Any]]:
        """
        Prepare user content for multimodal input.
        
        Args:
            content: List of mixed content items (strings, images, arrays)
            
        Returns:
            List[Dict]: Formatted content list for OpenAI API
        """
        formatted_content = []
        
        for item in content:
            if isinstance(item, str):
                # Text content
                formatted_content.append({
                    "type": "text",
                    "text": item
                })
            elif isinstance(item, (Image.Image, np.ndarray)):
                # Image content
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_base64_url(item)
                    }
                })
            else:
                # Convert other types to string
                formatted_content.append({
                    "type": "text",
                    "text": str(item)
                })
        
        return formatted_content
    
    def _build_custom_messages(self, request: LLMRequest) -> List[Dict[str, Any]]:
        """
        Build the complete message list for Gemini API request.
        
        Args:
            request: LLMRequest object containing the request data
            
        Returns:
            List[Dict]: Complete message list for Gemini API
        """
        # Gemini API expects messages in a different format
        user_content = self._prepare_custom_content(request.content)
        messages = [{"role": "user", "content": user_content}]
        return messages
    
    def _build_messages(self, request: LLMRequest) -> List[Dict[str, Any]]:
        """
        Build the complete message list for the API request.
        
        Args:
            request: LLMRequest object containing the request data
            
        Returns:
            List[Dict]: Complete message list for OpenAI API
        """
        messages = []
        
        # Add system prompt if provided
        system_message = self._prepare_system_prompt(request.system_prompt)
        if system_message:
            messages.append(system_message)
        
        # Add user content
        user_content = self._prepare_user_content(request.content)
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
    
    def send_request(
        self, 
        content: Union[List[Union[str, Image.Image, np.ndarray]], LLMRequest],
        system_prompt: Optional[str] = None,
        generation_params: Optional[Dict[str, Any]] = None,
        post_processor: Optional[Callable[[str], Any]] = None,
        timeout: Optional[int] = 5,
    ) -> Dict[str, Any]:
        """
        Send a request to the LLM server.
        
        Args:
            content: Either a list of mixed content items or an LLMRequest object
            system_prompt: Optional system prompt (ignored if content is LLMRequest)
            generation_params: Optional generation parameters
            post_processor: Optional post-processing function
            
        Returns:
            Dict containing:
                - 'response': Raw response text from the model
                - 'processed_response': Post-processed response (if post_processor provided)
                - 'success': Boolean indicating request success
                - 'error': Error message if request failed
                - 'metadata': Additional metadata (tokens, timing, etc.)
        """
        # Handle different input types
        if isinstance(content, LLMRequest):
            request = content
        else:
            request = LLMRequest(
                content=content,
                system_prompt=system_prompt,
                generation_params=generation_params
            )
        
        # Prepare generation parameters
        gen_params = self.default_generation_params.copy()
        if request.generation_params:
            gen_params.update(request.generation_params)
        
        # Build messages
        messages = self._build_messages(request)
        
        # Determine which post-processor to use
        processor = post_processor or self.post_processor
        
        # Attempt the request with retry logic
        last_exception = None
        
        # Handle different client types
        if self.client_type == "custom":
            return self._handle_custom_request(request, messages, gen_params, processor)
        else:
            return self._handle_openai_request(messages, gen_params, processor)
    
    def _handle_custom_request(self, request: LLMRequest, messages: List[Dict[str, Any]], gen_params: dict, processor) -> Dict[str, Any]:
        """Handle request using Custom client."""
        try:
            start_time = time.time()
            
            # Build Custom-specific messages
            gemini_messages = self._build_custom_messages(request)
            
            # Extract system prompt
            system_prompt = request.system_prompt or "You are a helpful assistant."
            
            # Make the request
            response, success = self.client.request(
                messages=gemini_messages,
                system=system_prompt,
                params=gen_params,
                timeout=300,
                max_retries=self.max_retries,
            )
            
            end_time = time.time()
            
            if success:
                # Extract response text from Custom format
                response_text = response.get("answer", [{}])[0].get("value", "")
                
                # Prepare result
                result = {
                    'response': response_text,
                    'success': True,
                    'error': None,
                    'metadata': {
                        'model': self.model_name,
                        'response_time': end_time - start_time,
                        'client_type': 'custom'
                    }
                }
                
                # Apply post-processing if available
                if processor:
                    try:
                        result['processed_response'] = processor(response_text)
                    except Exception as e:
                        result['processed_response'] = None
                        result['post_processing_error'] = str(e)
                else:
                    result['processed_response'] = response_text
                
                return result
            else:
                return {
                    'response': None,
                    'processed_response': None,
                    'success': False,
                    'error': response.get("error", "Custom request failed"),
                    'metadata': {
                        'model': self.model_name,
                        'client_type': 'custom',
                        'response_time': end_time - start_time
                    }
                }
                
        except Exception as e:
            return {
                'response': None,
                'processed_response': None,
                'success': False,
                'error': str(e),
                'metadata': {
                    'model': self.model_name,
                    'client_type': 'custom',
                    'final_error': str(e)
                }
            }
    
    def _handle_openai_request(self, messages: List[Dict[str, Any]], gen_params: dict, processor) -> Dict[str, Any]:
        """Handle request using OpenAI client."""
        last_exception = None
        
        # Handle OpenAI client (original logic)
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    timeout=300,
                    **gen_params
                )
                
                end_time = time.time()
                
                # Extract response text
                response_text = response.choices[0].message.content
                
                # Prepare result
                result = {
                    'response': response_text,
                    'success': True,
                    'error': None,
                    'metadata': {
                        'model': self.model_name,
                        'attempt': attempt + 1,
                        'response_time': end_time - start_time,
                        'usage': getattr(response, 'usage', None),
                        'finish_reason': response.choices[0].finish_reason,
                        'client_type': 'openai'
                    }
                }
                
                # Apply post-processing if available
                if processor:
                    try:
                        result['processed_response'] = processor(response_text)
                    except Exception as e:
                        result['processed_response'] = None
                        result['post_processing_error'] = str(e)
                else:
                    result['processed_response'] = response_text
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    # Final attempt failed
                    return {
                        'response': None,
                        'processed_response': None,
                        'success': False,
                        'error': str(last_exception),
                        'metadata': {
                            'model': self.model_name,
                            'attempts': self.max_retries,
                            'final_error': str(last_exception),
                            'client_type': 'openai'
                        }
                    }
    
    def set_post_processor(self, post_processor: Callable[[str], Any]):
        """
        Set the default post-processing function.
        
        Args:
            post_processor: Function that takes response text and returns processed result
        """
        self.post_processor = post_processor
    
    def update_generation_params(self, **params):
        """
        Update default generation parameters.
        
        Args:
            **params: Generation parameters to update
        """
        self.default_generation_params.update(params)


def extract_score_from_answer(text: str):
    pattern = r'score\s*[:=]\s*([+-]?\d*\.?\d+)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return -1
    return -1