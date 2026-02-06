"""
Anthropic代理服务器 - 将OpenAI格式请求转换为Anthropic格式
使用anthropic官方SDK
"""
import os
import json
import asyncio
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from anthropic import Anthropic, AsyncAnthropic

from config import API_KEY, MODEL_NAME, BASE_URL, HOST, PORT


app = FastAPI(
    title="Anthropic Proxy",
    description="将OpenAI格式请求转换为Anthropic格式的代理服务器",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    model: Optional[str] = MODEL_NAME
    messages: list
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    system: Optional[str] = None


# 创建Anthropic客户端
def get_anthropic_client():
    """获取Anthropic客户端"""
    return AsyncAnthropic(
        api_key=API_KEY,
        base_url=BASE_URL,
    )


def convert_openai_to_anthropoc_messages(messages: list) -> list:
    """转换OpenAI消息格式为Anthropic格式"""
    anthropic_messages = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        # 跳过system消息（Anthropic单独处理）
        if role == "system":
            continue

        # 转换content为Anthropic格式
        if isinstance(content, str):
            anthropic_content = content
        elif isinstance(content, list):
            # 处理多模态内容
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    text_parts.append(f"[Image: {item.get('image_url', {}).get('url', '')}]")
            anthropic_content = "\n".join(text_parts) if text_parts else ""
        else:
            anthropic_content = str(content)

        anthropic_messages.append({
            "role": role,
            "content": anthropic_content
        })

    return anthropic_messages


def convert_anthropic_to_openai_response(response, model: str) -> dict:
    """转换Anthropic响应为OpenAI格式"""
    # 提取文本内容
    text_content = ""
    if response.content:
        for block in response.content:
            if block.type == "text":
                text_content = block.text
                break

    return {
        "id": response.id or f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text_content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": response.usage.input_tokens if response.usage else 0,
            "completion_tokens": response.usage.output_tokens if response.usage else 0,
            "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0
        }
    }


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Anthropic Proxy Server",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "health": "/health",
            "models": "/v1/models"
        }
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": 1700000000,
            "owned_by": "anthropic-proxy"
        }]
    }


async def stream_generator(client, request: ChatRequest):
    """流式响应生成器"""
    try:
        # 转换消息格式
        anthropic_messages = convert_openai_to_anthropoc_messages(request.messages)

        # 构建请求参数
        kwargs = {
            "model": request.model or MODEL_NAME,
            "max_tokens": request.max_tokens or 4096,
            "messages": anthropic_messages,
        }

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.system:
            kwargs["system"] = request.system

        chunk_id = f"chatcmpl-{int(time.time())}"
        model_name = request.model or MODEL_NAME

        # 调用流式API
        async with client.messages.stream(**kwargs) as stream:
            # 发送初始chunk (role)
            yield f"data: {json.dumps({
                'id': chunk_id,
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': model_name,
                'choices': [{
                    'index': 0,
                    'delta': {'role': 'assistant'},
                    'finish_reason': None
                }]
            }, ensure_ascii=False)}\n\n"

            # 流式传输文本
            async for text in stream.text_stream:
                yield f"data: {json.dumps({
                    'id': chunk_id,
                    'object': 'chat.completion.chunk',
                    'created': int(time.time()),
                    'model': model_name,
                    'choices': [{
                        'index': 0,
                        'delta': {'content': text},
                        'finish_reason': None
                    }]
                }, ensure_ascii=False)}\n\n"

            # 发送结束chunk
            yield f"data: {json.dumps({
                'id': chunk_id,
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': model_name,
                'choices': [{
                    'index': 0,
                    'delta': {},
                    'finish_reason': 'stop'
                }]
            }, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': f'{type(e).__name__}: {str(e)}'})}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """聊天完成接口（支持流式和非流式）"""
    client = get_anthropic_client()

    # 流式请求
    if request.stream:
        return StreamingResponse(
            stream_generator(client, request),
            media_type="text/event-stream"
        )

    # 非流式请求
    try:
        # 转换消息格式
        anthropic_messages = convert_openai_to_anthropoc_messages(request.messages)

        # 构建请求参数
        kwargs = {
            "model": request.model or MODEL_NAME,
            "max_tokens": request.max_tokens or 4096,
            "messages": anthropic_messages,
        }

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.system:
            kwargs["system"] = request.system

        # 调用Anthropic API
        response = await client.messages.create(**kwargs)

        # 转换响应格式
        openai_response = convert_anthropic_to_openai_response(
            response,
            request.model or MODEL_NAME
        )

        return openai_response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Anthropic API error: {type(e).__name__}: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║           Anthropic Proxy Server                              ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  API Key: {API_KEY[:20]}...                            ║
    ║  Model: {MODEL_NAME}                              ║
    ║  Base URL: {BASE_URL}              ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Server: http://{HOST}:{PORT}                                ║
    ║  API: http://localhost:{PORT}/v1/chat/completions              ║
    ║  Docs: http://localhost:{PORT}/docs                           ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(app, host=HOST, port=PORT)
