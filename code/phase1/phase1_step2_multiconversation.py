"""
Phase1 Step2: 多轮对话与上下文管理
学习目标：理解LLM是无状态的，需要客户端维护消息历史
"""

from dotenv import load_dotenv
import os
from dashscope import Generation

load_dotenv()
import dashscope
dashscope.api_key = os.getenv("ANTHROPIC_API_KEY")

def simple_chat():
    """简单的单轮对话"""
    print("=== 单轮对话示例 ===")
    
    response = Generation.call(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "你是一个助手。"},
            {"role": "user", "content": "我叫张三，今年25岁"}
        ],
        temperature=0.3
    )
    
    if response.status_code == 200:
        print(f"AI: {response.output.text}")
        
        # 第二轮对话，不传递历史
        response2 = Generation.call(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是一个助手。"},
                {"role": "user", "content": "我今年多大？"}
            ],
            temperature=0.3
        )
        
        if response2.status_code == 200:
            print(f"AI: {response2.output.text}")
            print("注意：AI不知道你刚才说过的信息")

def multi_round_chat():
    """多轮对话，维护上下文"""
    print("\n=== 多轮对话示例 ===")
    
    # 维护消息历史
    messages = [
        {"role": "system", "content": "你是一个助手，请记住用户的信息。"}
    ]
    
    # 第一轮
    user_input1 = "我叫李四，今年30岁，住在北京"
    messages.append({"role": "user", "content": user_input1})
    
    response1 = Generation.call(
        model="qwen-plus",
        messages=messages,
        temperature=0.3
    )
    
    if response1.status_code == 200:
        ai_reply1 = response1.output.text
        print(f"用户: {user_input1}")
        print(f"AI: {ai_reply1}")
        
        # 把AI的回复也加入历史
        messages.append({"role": "assistant", "content": ai_reply1})
        
        # 第二轮
        user_input2 = "我今年多大？住在哪里？"
        messages.append({"role": "user", "content": user_input2})
        
        response2 = Generation.call(
            model="qwen-plus",
            messages=messages,
            temperature=0.3
        )
        
        if response2.status_code == 200:
            ai_reply2 = response2.output.text
            print(f"\n用户: {user_input2}")
            print(f"AI: {ai_reply2}")
            print("注意：AI记住了之前的信息！")

def interactive_chat():
    """交互式对话程序"""
    print("\n=== 交互式对话程序 ===")
    print("输入'quit'退出对话")
    
    # 初始化消息历史
    messages = [
        {"role": "system", "content": "你是一个友好的助手，请记住对话中的所有信息。"}
    ]
    
    round_count = 0
    
    while True:
        round_count += 1
        user_input = input(f"\n第{round_count}轮 - 你: ")
        
        if user_input.lower() == 'quit':
            print("再见！")
            break
        
        # 添加用户消息
        messages.append({"role": "user", "content": user_input})
        
        # 调用API
        response = Generation.call(
            model="qwen-plus",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        
        if response.status_code == 200:
            ai_reply = response.output.text
            print(f"AI: {ai_reply}")
            
            # 添加AI回复到历史
            messages.append({"role": "assistant", "content": ai_reply})
            
            # 显示当前对话轮数和token使用情况
            print(f"(已对话{round_count}轮，当前消息数: {len(messages)})")
            
            if hasattr(response, 'usage'):
                usage = response.usage
                print(f"本次调用Token: {usage}")
        else:
            print(f"API调用失败: {response.code} - {response.message}")

def test_memory():
    """测试AI的记忆能力"""
    print("\n=== 记忆能力测试 ===")
    
    messages = [
        {"role": "system", "content": "你是一个助手，请记住用户提到的所有信息。"}
    ]
    
    # 前几轮对话
    conversations = [
        "我喜欢编程，特别是Python",
        "我有一只猫，叫小花",
        "我住在上海"
    ]
    
    for i, msg in enumerate(conversations, 1):
        messages.append({"role": "user", "content": msg})
        
        response = Generation.call(
            model="qwen-plus",
            messages=messages,
            temperature=0.3
        )
        
        if response.status_code == 200:
            ai_reply = response.output.text
            messages.append({"role": "assistant", "content": ai_reply})
            print(f"第{i}轮 - 用户: {msg}")
            print(f"第{i}轮 - AI: {ai_reply}")
    
    # 测试记忆
    test_questions = [
        "我喜欢什么编程语言？",
        "我的猫叫什么名字？", 
        "我住在哪里？"
    ]
    
    print(f"\n=== 记忆测试 ===")
    for i, question in enumerate(test_questions, 1):
        messages.append({"role": "user", "content": question})
        
        response = Generation.call(
            model="qwen-plus",
            messages=messages,
            temperature=0.3
        )
        
        if response.status_code == 200:
            ai_reply = response.output.text
            messages.append({"role": "assistant", "content": ai_reply})
            print(f"测试{i} - 问题: {question}")
            print(f"测试{i} - 回答: {ai_reply}")

def demonstrate_context_limit():
    """演示上下文窗口限制"""
    print("\n=== 上下文窗口限制演示 ===")
    
    messages = [
        {"role": "system", "content": "你是一个助手。"}
    ]
    
    # 模拟长对话
    print("模拟长对话，观察token使用...")
    
    for i in range(10):
        user_msg = f"这是第{i+1}轮对话，请告诉我一些关于Python的知识。"
        messages.append({"role": "user", "content": user_msg})
        
        response = Generation.call(
            model="qwen-plus",
            messages=messages,
            temperature=0.3,
            max_tokens=100
        )
        
        if response.status_code == 200:
            ai_reply = response.output.text
            messages.append({"role": "assistant", "content": ai_reply})
            
            if hasattr(response, 'usage'):
                usage = response.usage
                print(f"第{i+1}轮 - 总Token: {usage.get('total_tokens', 0)}")
                
                # 检查是否接近限制（假设限制是8000 tokens）
                total_tokens = usage.get('total_tokens', 0)
                if total_tokens > 7000:
                    print(f"警告: 接近上下文限制 ({total_tokens} tokens)")
                    break

if __name__ == "__main__":
    print("=== Phase1 Step2: 多轮对话与上下文管理 ===")
    
    # 运行所有示例
    simple_chat()
    multi_round_chat()
    
    print("\n" + "="*50)
    print("选择要运行的示例:")
    print("1. 交互式对话")
    print("2. 记忆能力测试") 
    print("3. 上下文限制演示")
    print("4. 跳过所有演示")
    
    choice = input("\n请选择 (1-4): ")
    
    if choice == "1":
        interactive_chat()
    elif choice == "2":
        test_memory()
    elif choice == "3":
        demonstrate_context_limit()
    else:
        print("演示完成！")
