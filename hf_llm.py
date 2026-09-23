"""
HuggingFace LLM Interface for RLVR.

Integrates Unsloth-optimized open-source models (like Qwen2.5) with our
Multi-Turn MDP. Implements the LLMInterface protocol.
"""

import threading
import torch
from typing import List, Dict
from transformers import PreTrainedModel, PreTrainedTokenizer

class HuggingFaceLLM:
    """
    Real language model implementation for the RLVR MDP loop.
    
    Wraps a HuggingFace/Unsloth model to generate code and reasoning
    in response to the MDP's conversation history.
    """
    
    def __init__(
        self, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer,
        max_new_tokens: int = 512,
        temperature: float = 0.9,
        top_p: float = 0.95
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.gen_lock = threading.Lock()
        
        # Ensure we're in inference mode for MDP generation
        self.model.eval()

    @torch.no_grad()
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response given a conversation history.
        
        Args:
            messages: List of dicts with 'role' and 'content'.
            
        Returns:
            The generated response string.
        """
        # Apply the chat template (Qwen2.5 / ChatML format)
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate completion (locked for thread safety with Accelerate hooks)
        with self.gen_lock:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                max_length=None,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True  # Important for fast inference during RL
            )
        
        # Extract only the newly generated tokens
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        
        response = self.tokenizer.decode(
            generated_tokens, 
            skip_special_tokens=True
        )
        
        return response.strip()
