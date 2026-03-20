"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import re
import time
import logging
from typing import Optional, Dict, Any, List
from openai import OpenAI, RateLimitError, APIError

from ..config import Config

logger = logging.getLogger('mirofish.llm_client')


class LLMClient:
    """LLM客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）
            
        Returns:
            模型响应文本
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        # Retry with exponential backoff on rate limit errors
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                # 部分模型（如MiniMax M2.5）会在content中包含<think>思考内容，需要移除
                content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
                return content
            except RateLimitError as e:
                # Parse retry-after from error message if available
                wait_time = min(30 * (2 ** attempt), 120)  # 30s, 60s, 120s max
                error_msg = str(e)
                # Try to extract wait time from "Please try again in Xm Ys" pattern
                import re as re_mod
                match = re_mod.search(r'try again in (\d+)m', error_msg)
                if match:
                    wait_time = min(int(match.group(1)) * 60, 300)  # Cap at 5 min
                else:
                    match = re_mod.search(r'try again in (\d+\.?\d*)s', error_msg)
                    if match:
                        wait_time = min(int(float(match.group(1))) + 1, 120)

                if attempt < max_retries - 1:
                    logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt+2}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    raise
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            解析后的JSON对象
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # 清理markdown代码块标记
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned_response}")

