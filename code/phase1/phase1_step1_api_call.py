"""
API调用方法汇总
包含3种不同的调用方式：requests、OpenAI兼容格式、官方SDK
"""

from dotenv import load_dotenv
import os
import requests
import dashscope
from openai import OpenAI
from dashscope import Generation


load_dotenv()

def call_with_requests():
    """方法1：使用requests直接调用HTTP API"""
    print("\n=== 方法1：requests ===")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "qwen-plus",
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个简洁的助手，回答不超过50字。"
                },
                {
                    "role": "user", 
                    "content": "Python 中列表和元组有什么区别？"
                }
            ]
        },
        "parameters": {
            "temperature": 0.3,
            "max_tokens": 200
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            if "output" in result and "text" in result["output"]:
                answer = result["output"]["text"]
                print(f"requests调用成功!")
                print(f"回答: {answer}")
                
                if "usage" in result:
                    usage = result["usage"]
                    print(f"Token使用: {usage}")
                
                return answer
            else:
                print(f"响应格式异常: {result}")
        else:
            print(f"API调用失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"调用异常: {e}")
    
    return None

def call_with_openai_format():
    """方法2：使用OpenAI兼容格式"""
    print("\n=== 方法2：OpenAI兼容格式 ===")
    
    client = OpenAI(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {
                    "role": "system", 
                    "content": "你是一个简洁的助手，回答不超过50字。"
                },
                {
                    "role": "user", 
                    "content": "Python 中列表和元组有什么区别？"
                }
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        answer = response.choices[0].message.content
        print(f"OpenAI兼容格式调用成功!")
        print(f"回答: {answer}")
        
        if hasattr(response, 'usage'):
            usage = response.usage
            print(f"Token使用: {usage}")
        
        return answer
        
    except Exception as e:
        print(f"调用失败: {e}")
        return None

def call_with_official_sdk():
    """方法3：使用阿里云官方SDK"""
    print("\n=== 方法3：官方SDK ===")

    dashscope.api_key = os.getenv("ANTHROPIC_API_KEY")
    
    try:
        response = Generation.call(
            model="qwen-plus",
            messages=[
                {
                    "role": "system", 
                    "content": "你是一个简洁的助手，回答不超过50字。"
                },
                {
                    "role": "user", 
                    "content": "Python 中列表和元组有什么区别？"
                }
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        if response.status_code == 200:
            answer = response.output.text
            print(f"官方SDK调用成功!")
            print(f"回答: {answer}")
            
            if hasattr(response, 'usage'):
                usage = response.usage
                print(f"Token使用: {usage}")
            
            return answer
        else:
            print(f"调用失败: {response.code} - {response.message}")
            return None
            
    except Exception as e:
        print(f"调用异常: {e}")
        return None

def test_all_methods():
    """测试所有三种方法"""
    print("=== API调用方法对比测试 ===")
    
    # 测试所有方法
    methods = [
        call_with_requests,
        call_with_openai_format,
        call_with_official_sdk
    ]
    
    for i, method in enumerate(methods, 1):
        print(f"\n{'='*50}")
        print(f"测试方法 {i}: {method.__name__}")
        print('='*50)
        
        result = method()
        
        if result:
            print(f"方法{i}成功")
        else:
            print(f"方法{i}失败")

if __name__ == "__main__":
    # 可以单独测试某个方法
    # call_with_requests()
    # call_with_openai_format()
    # call_with_official_sdk()
    
    # 或者测试所有方法
    test_all_methods()
