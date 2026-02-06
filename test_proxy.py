"""
测试脚本 - 验证Anthropic代理功能
"""
import asyncio
import httpx
import json
from config import API_KEY, MODEL_NAME, BASE_URL, HOST, PORT


# 代理服务器地址
PROXY_URL = f"http://localhost:{PORT}/v1/chat/completions"


async def test_direct_anthropic():
    """直接测试Anthropic API"""
    print("\n" + "="*60)
    print("测试 1: 直接调用 Anthropic API")
    print("="*60)

    from anthropic import Anthropic

    client = Anthropic(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "你好，你是谁？请简短回答。"}
            ]
        )
        print(f"状态: 成功")
        print(f"响应: {response.content[0].text}")
        return True
    except Exception as e:
        print(f"状态: 失败")
        print(f"错误: {type(e).__name__}: {e}")
        return False


async def test_proxy_non_stream():
    """测试代理 - 非流式请求"""
    print("\n" + "="*60)
    print("测试 2: 代理服务器 - 非流式请求 (OpenAI格式)")
    print("="*60)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "你好，你是谁？请简短回答。"}
        ],
        "max_tokens": 1024
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                PROXY_URL,
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                data = response.json()
                print(f"状态: 成功")
                print(f"响应ID: {data.get('id')}")
                print(f"模型: {data.get('model')}")
                print(f"内容: {data['choices'][0]['message']['content']}")
                print(f"Usage: {data.get('usage')}")
                return True
            else:
                print(f"状态: 失败 (HTTP {response.status_code})")
                print(f"响应: {response.text}")
                return False

    except Exception as e:
        print(f"状态: 失败")
        print(f"错误: {type(e).__name__}: {e}")
        return False


async def test_proxy_with_system():
    """测试代理 - 带system消息"""
    print("\n" + "="*60)
    print("测试 3: 代理服务器 - 带System消息")
    print("="*60)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个有帮助的AI助手，请用中文回答。"},
            {"role": "user", "content": "1+1等于几？"}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                PROXY_URL,
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                data = response.json()
                print(f"状态: 成功")
                print(f"内容: {data['choices'][0]['message']['content']}")
                return True
            else:
                print(f"状态: 失败 (HTTP {response.status_code})")
                print(f"响应: {response.text}")
                return False

    except Exception as e:
        print(f"状态: 失败")
        print(f"错误: {type(e).__name__}: {e}")
        return False


async def test_proxy_stream():
    """测试代理 - 流式请求"""
    print("\n" + "="*60)
    print("测试 4: 代理服务器 - 流式请求")
    print("="*60)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "用一句话介绍Python编程语言。"}
        ],
        "max_tokens": 512,
        "stream": True
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{PROXY_URL}/stream",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                print(f"状态: {'成功' if response.status_code == 200 else f'失败 (HTTP {response.status_code})'}")

                full_content = ""
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            if "choices" in chunk and chunk["choices"]:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_content += content
                                    print(content, end="", flush=True)
                        except json.JSONDecodeError:
                            pass

                print(f"\n完整内容: {full_content}")
                return True

    except Exception as e:
        print(f"状态: 失败")
        print(f"错误: {type(e).__name__}: {e}")
        return False


async def test_models_endpoint():
    """测试模型列表接口"""
    print("\n" + "="*60)
    print("测试 5: 模型列表接口")
    print("="*60)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"http://localhost:{PORT}/v1/models")

            if response.status_code == 200:
                data = response.json()
                print(f"状态: 成功")
                print(f"模型列表: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return True
            else:
                print(f"状态: 失败 (HTTP {response.status_code})")
                return False

    except Exception as e:
        print(f"状态: 失败")
        print(f"错误: {type(e).__name__}: {e}")
        return False


async def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("Anthropic 代理服务器测试套件")
    print("="*60)
    print(f"代理地址: http://localhost:{PORT}")
    print(f"上游API: {BASE_URL}")
    print(f"模型: {MODEL_NAME}")

    # 测试结果
    results = {}

    # 测试1: 直接API调用
    results["direct"] = await test_direct_anthropic()

    # 等待用户启动代理服务器
    input("\n请先启动代理服务器 (python main.py)，然后按回车继续...")

    # 测试2: 非流式请求
    results["non_stream"] = await test_proxy_non_stream()

    # 测试3: 带system消息
    results["with_system"] = await test_proxy_with_system()

    # 测试4: 流式请求
    results["stream"] = await test_proxy_stream()

    # 测试5: 模型列表
    results["models"] = await test_models_endpoint()

    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    for name, passed in results.items():
        status = "通过" if passed else "失败"
        print(f"{name:20s}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n总计: {passed}/{total} 测试通过")


if __name__ == "__main__":
    asyncio.run(main())
