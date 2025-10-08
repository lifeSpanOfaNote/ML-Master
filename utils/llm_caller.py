import os
from typing import List, Dict, Union, Optional, Any
from openai import OpenAI
import time
import logging
logger = logging.getLogger("ml-master")


class LLM:
    """
    Encapsulate the VLLM-based LLM class to invoke the self-hosted VLLM model via the OpenAI SDK.
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy-key",
        model_name: str = "default-model",
        temperature: float = 0.7,
        max_tokens: int = 16384,
        stop_tokens: Optional[Union[str, List[str]]] = None,
        retry_time: int = 20,
        delay_time: int = 3,
    ):
        """
        Initialize the VLLM LLM class.
        
        Args:
            base_url: The URL of the VLLM service.
            api_key: API key (generally not important when self-hosted).
            model_name: Name of the model (generally not important when self-hosted).
            temperature: Temperature parameter to control output randomness.
            max_tokens: Maximum number of tokens to generate.
            stop_tokens: List of tokens that signal the end of generation.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop_tokens = stop_tokens
        self.retry_time = retry_time
        self.delay_time = delay_time
        
        # initalize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def generate(
        self, 
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_tokens: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Any]:
        """
        Generate text
        
        Args:
            messages: List of conversation messages
            temperature: Overrides the default temperature parameter
            max_tokens: Overrides the default maximum number of tokens
            stop_tokens: Overrides the default stop sequences
            stream: Whether to use streaming output
            **kwargs: Additional parameters passed to the OpenAI API
            
        Returns:
            If stream=False, returns the generated text  
            If stream=True, returns the streaming response object
        """
        
        # use function parameters or default values
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        stops = stop_tokens if stop_tokens is not None else self.stop_tokens
        
        # create request parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
            "stream": stream,
            **kwargs
        }
        
        # add stop_tokens
        if stops is not None:
            params["stop"] = stops
            
        attempt = 0
        while attempt < self.retry_time:
            try:
                response = self.client.chat.completions.create(**params)
                
                if stream:
                    return response
                
                return response.choices[0].message
            except Exception as e:
                attempt += 1
                logger.warning(f"calling llm failed, retrying {attempt}/retry, error message: {e}")
                if attempt >= self.retry_time:
                    logger.error("LLM call retry limit reached, throwing exception")
                    raise e
                time.sleep(self.delay_time)

    def stream_generate(
        self, 
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_tokens: Optional[Union[str, List[str]]] = None,
        delay_time:int = 1,
        **kwargs
    ) -> Union[str, Any]:
        """
        Generate text
        
        Args:
            messages: List of conversation messages
            temperature: Overrides the default temperature parameter
            max_tokens: Overrides the default maximum number of tokens
            stop_tokens: Overrides the default stop sequences
            stream: Whether to use streaming output
            **kwargs: Additional parameters passed to the OpenAI API
            
        Returns:
            If stream=False, returns the generated text  
            If stream=True, returns the streaming response object
        """
        # use function parameters or default values
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        stops = stop_tokens if stop_tokens is not None else self.stop_tokens
        stream = True
        
        # create request parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
            "stream": stream,
            **kwargs
        }
        
        # add stop_tokens
        if stops is not None:
            params["stop"] = stops
        attempt = 0
        while attempt < self.retry_time:
            try:
                response = self.client.chat.completions.create(**params)
                full_text = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        full_text += chunk.choices[0].delta.content
                    return response
            except Exception as e:
                attempt += 1
                logger.warning(f"calling llm failed, retrying {attempt}/retry, error message: {e}")
                if attempt >= self.retry_time:
                    logger.error("LLM call retry limit reached, throwing exception")
                    raise e
                time.sleep(self.delay_time)
                
    
    def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_tokens: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Any]:
        """
        Text Completion

        Args:
            prompt: Text prompt
            temperature: Overrides the default temperature parameter
            max_tokens: Overrides the default maximum number of tokens
            stop_tokens: Overrides the default stop tokens
            stream: Whether to use streaming output
            **kwargs: Additional parameters passed to the OpenAI API

        Returns:
            If stream=False, returns the generated text  
            If stream=True, returns a streaming response object
        """

        # use function parameters or default values
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        stops = stop_tokens if stop_tokens is not None else self.stop_tokens
        
        # create request parameters
        params = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temp,
            "max_tokens": tokens,
            "stream": stream,
            **kwargs
        }
        
        # add stop_tokens
        if stops is not None:
            params["stop"] = stops
            
        response = self.client.completions.create(**params)
        
        if stream:
            return response
        
        return response.choices[0].text
    
    def stream_complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_tokens: Optional[Union[str, List[str]]] = None,
        return_metrics: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Text Completion with optional metrics

        Args:
            prompt: Text prompt
            temperature: Overrides the default temperature parameter
            max_tokens: Overrides the default maximum number of tokens
            stop_tokens: Overrides the default stop tokens
            return_metrics: If True, return dict with text and metrics; if False, return text only
            **kwargs: Additional parameters passed to the OpenAI API

        Returns:
            If return_metrics=False: returns the generated text string
            If return_metrics=True: returns dict with:
                - text: generated text
                - input_tokens: number of input tokens
                - output_tokens: number of output tokens  
                - response_time: response time in seconds
                - total_tokens: total tokens used
        """

        # use function parameters or default values
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        stops = stop_tokens if stop_tokens is not None else self.stop_tokens
        stream = True
        # create request parameters
        params = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temp,
            "max_tokens": tokens,
            "stream": stream,
            "stream_options": {"include_usage": True},  # Request usage info in streaming
            **kwargs
        }
        
        # add stop_tokens
        if stops is not None:
            params["stop"] = stops
        
        attempt = 0
        while attempt < self.retry_time:
            try:
                start_time = time.time()
                response = self.client.completions.create(**params)
                
                full_text = ""
                usage_info = None
                
                for chunk in response:
                    if chunk.choices and chunk.choices[0].text is not None:
                        full_text += chunk.choices[0].text
                    # Capture usage information from the last chunk
                    if hasattr(chunk, 'usage') and chunk.usage is not None:
                        usage_info = chunk.usage
                
                response_time = time.time() - start_time
                
                if return_metrics:
                    # Extract token counts from usage info
                    input_tokens = usage_info.prompt_tokens if usage_info else 0
                    output_tokens = usage_info.completion_tokens if usage_info else 0
                    total_tokens = usage_info.total_tokens if usage_info else 0
                    
                    return {
                        "text": full_text,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "response_time": response_time,
                    }
                else:
                    return full_text
                    
            except Exception as e:
                attempt += 1
                logger.warning(f"calling llm failed, retrying {attempt}/retry, error message: {e}")
                if attempt >= self.retry_time:
                    logger.error("LLM call retry limit reached, throwing exception")
                    raise e
                time.sleep(self.delay_time)
    
