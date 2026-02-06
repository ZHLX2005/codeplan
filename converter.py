"""
OpenAI <-> Anthropic 格式转换器
"""
from typing import Dict, List, Any, Optional
from datetime import datetime


class OpenAIToAnthropicConverter:
    """OpenAI请求转Anthropic格式"""

    @staticmethod
    def convert_request(openai_request: Dict[str, Any]) -> Dict[str, Any]:
        """转换OpenAI请求为Anthropic格式"""
        anthropic_request = {
            "model": openai_request.get("model", "claude-3-sonnet-20240229"),
            "max_tokens": openai_request.get("max_tokens", 4096),
            "messages": OpenAIToAnthropicConverter._convert_messages(openai_request.get("messages", []))
        }

        # 可选参数
        if "temperature" in openai_request:
            anthropic_request["temperature"] = openai_request["temperature"]
        if "top_p" in openai_request:
            anthropic_request["top_p"] = openai_request["top_p"]
        if "stream" in openai_request:
            anthropic_request["stream"] = openai_request["stream"]
        if "system" in openai_request:
            anthropic_request["system"] = openai_request["system"]

        return anthropic_request

    @staticmethod
    def _convert_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """转换消息格式"""
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # 跳过system消息（Anthropic不放在messages中）
            if role == "system":
                continue

            # 转换content为Anthropic格式
            if isinstance(content, str):
                anthropic_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                anthropic_content = []
                for item in content:
                    if item.get("type") == "text":
                        anthropic_content.append({
                            "type": "text",
                            "text": item.get("text", "")
                        })
                    elif item.get("type") == "image_url":
                        # 简化处理图片
                        anthropic_content.append({
                            "type": "text",
                            "text": f"[Image: {item.get('image_url', {}).get('url', '')}]"
                        })
                if not anthropic_content:
                    anthropic_content = [{"type": "text", "text": ""}]
            else:
                anthropic_content = [{"type": "text", "text": str(content)}]

            anthropic_messages.append({
                "role": role,
                "content": anthropic_content
            })

        return anthropic_messages


class AnthropicToOpenAIConverter:
    """Anthropic响应转OpenAI格式"""

    @staticmethod
    def convert_response(anthropic_response: Dict[str, Any], model: str) -> Dict[str, Any]:
        """转换Anthropic响应为OpenAI格式"""
        # 提取文本内容
        content = anthropic_response.get("content", [])
        text_content = ""
        if content and isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    text_content = item.get("text", "")
                    break

        openai_response = {
            "id": anthropic_response.get("id", f"chatcmpl-{datetime.now().timestamp()}"),
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text_content
                },
                "finish_reason": AnthropicToOpenAIConverter._convert_stop_reason(
                    anthropic_response.get("stop_reason", "end_turn")
                )
            }],
            "usage": {
                "prompt_tokens": anthropic_response.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": anthropic_response.get("usage", {}).get("output_tokens", 0),
                "total_tokens": (
                    anthropic_response.get("usage", {}).get("input_tokens", 0) +
                    anthropic_response.get("usage", {}).get("output_tokens", 0)
                )
            }
        }

        return openai_response

    @staticmethod
    def _convert_stop_reason(stop_reason: str) -> str:
        """转换停止原因"""
        mapping = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop"
        }
        return mapping.get(stop_reason, "stop")

    @staticmethod
    def convert_stream_chunk(chunk: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
        """转换流式响应块"""
        chunk_type = chunk.get("type")

        if chunk_type == "content_block_delta":
            return {
                "id": f"chatcmpl-{datetime.now().timestamp()}",
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": chunk.get("delta", {}).get("text", "")
                    },
                    "finish_reason": None
                }]
            }
        elif chunk_type == "message_stop":
            return {
                "id": f"chatcmpl-{datetime.now().timestamp()}",
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
        elif chunk_type == "message_start":
            return {
                "id": f"chatcmpl-{datetime.now().timestamp()}",
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant"
                    },
                    "finish_reason": None
                }]
            }

        return None
