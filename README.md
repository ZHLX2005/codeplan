# Anthropic Proxy

将OpenAI格式请求转换为Anthropic格式的代理服务器。

## 功能特点

- 兼容OpenAI API格式
- 支持非流式和流式响应
- 简单配置，开箱即用
- 支持豆包、智谱AI等兼容Anthropic接口的服务

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并修改配置：

```bash
cp .env.example .env
```

### 3. 启动服务

```bash
python main.py
```

服务将在 http://localhost:8080 启动

## API使用

### 聊天完成

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "doubao-seed-code-preview-latest",
    "messages": [
      {"role": "user", "content": "你好，你是谁？"}
    ]
  }'
```

### 使用OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="any-key"  # 可以是任意值
)

response = client.chat.completions.create(
    model="doubao-seed-code-preview-latest",
    messages=[
        {"role": "user", "content": "你好，你是谁？"}
    ]
)

print(response.choices[0].message.content)
```

## 配置说明

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| API_KEY | Anthropic API密钥 | - |
| MODEL_NAME | 模型名称 | doubao-seed-code-preview-latest |
| BASE_URL | API基础URL | https://ark.cn-beijing.volces.com/api/coding |
| HOST | 服务器主机 | 0.0.0.0 |
| PORT | 服务器端口 | 8080 |

## 端点

- `GET /` - 服务器信息
- `GET /health` - 健康检查
- `GET /v1/models` - 列出可用模型
- `POST /v1/chat/completions` - 聊天完成
- `POST /v1/chat/completions/stream` - 聊天完成（流式）
- `GET /docs` - API文档
