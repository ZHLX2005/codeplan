"""
配置文件
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API配置 - 请在.env文件中设置
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "claude-3-sonnet-20240229")
BASE_URL = os.getenv("BASE_URL", "https://api.anthropic.com")

# 服务器配置
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
