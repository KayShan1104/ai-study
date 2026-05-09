"""
Step 4: Function Calling / Tool Use
理解 LLM 如何决定调用外部工具，以及客户端如何执行并回传结果

完整流程:
1. 定义工具（JSON Schema）→ 告诉模型有哪些工具可用
2. 用户提问 → 模型分析意图，决定调用哪个工具、传什么参数
3. 模型返回 tool_calls（注意：模型不执行！）
4. 你的代码执行真正的工具函数
5. 把执行结果传回模型
6. 模型基于结果生成最终回复
"""

from dotenv import load_dotenv
import os
import json
from openai import OpenAI

load_dotenv()

# ============================================================
# 第 1 步：定义真正的工具函数 —— 这些是你的代码，模型看不到实现
# ============================================================

def get_weather(city: str) -> dict:
    """模拟天气查询，不需要接真实 API"""
    mock_data = {
        "北京": {"weather": "晴天", "temperature": 25, "wind": "微风"},
        "上海": {"weather": "多云", "temperature": 22, "wind": "东风 3级"},
        "广州": {"weather": "雷阵雨", "temperature": 30, "wind": "南风 4级"},
        "深圳": {"weather": "阴天", "temperature": 28, "wind": "无风"},
        "杭州": {"weather": "小雨", "temperature": 20, "wind": "北风 2级"},
    }
    if city in mock_data:
        return {"city": city, **mock_data[city]}
    return {"city": city, "error": f"未找到 {city} 的天气数据"}


def calculate(expression: str) -> dict:
    """用 eval 做简单数学计算"""
    try:
        # 只允许数字和运算符，防止任意代码执行
        if not all(c in "0123456789+-*/.() " for c in expression):
            return {"error": "表达式包含非法字符"}
        result = eval(expression)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 第 2 步：定义工具的 JSON Schema —— 这是模型的"菜单"
# ============================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息，包括温度、天气状况和风力",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海、广州"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式的值，支持加减乘除和小数",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如：1+2*3、100/7"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# 函数名 → 实际函数的映射，用于动态调用
function_map = {
    "get_weather": get_weather,
    "calculate": calculate,
}

# ============================================================
# 第 3 步：核心流程 —— 一次完整的 Function Calling
# ============================================================

def run_function_call(user_input: str):
    """
    跑通一次完整的 Function Calling 流程。
    核心是用 while 循环：只要模型返回 tool_calls，就执行工具并回传，
    直到模型不再需要调工具为止。这支持多轮连续调用的场景。
    """
    client = OpenAI(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    print(f"\n{'='*60}")
    print(f"用户提问: {user_input}")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": user_input}]

    round_num = 0
    total_usage = None

    while True:
        round_num += 1
        print(f"\n--- 第 {round_num} 轮 API 调用 ---")

        response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            tools=tools,
            temperature=0  # Function Calling 调试时设为 0，减少随机性
        )

        assistant_msg = response.choices[0].message
        total_usage = response.usage

        # 模型有两种回应：
        # A. tool_calls：决定调用工具 → 执行后继续循环
        # B. 普通文字：直接回复 → 跳出循环，结束
        if assistant_msg.tool_calls:
            print(f"模型决定调用 {len(assistant_msg.tool_calls)} 个工具:")

            # 把模型的决定追加到消息历史
            messages.append(assistant_msg)

            # 逐个执行模型决定的工具调用
            for tc in assistant_msg.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)

                print(f"  -> {func_name}({func_args})")

                # ★ 关键：这里是你的代码在执行，不是模型！
                func = function_map[func_name]
                result = func(**func_args)

                print(f"     执行结果: {result}")

                # 把工具结果作为 tool 角色消息传回给模型
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False)
                })

            # 本轮结束，进入下一轮，看模型是否还需要继续调工具
            print(f"  当前消息历史: {len(messages)} 条，进入下一轮...")

        else:
            # 模型直接文字回复，不需要再调工具
            print(f"模型直接文字回复: {assistant_msg.content}")
            break

    print(f"\n[Token 消耗]: prompt={total_usage.prompt_tokens}, "
          f"completion={total_usage.completion_tokens}, total={total_usage.total_tokens}")
    print(f"{'='*60}\n")


# ============================================================
# 第 4 步：测试不同场景
# ============================================================

if __name__ == "__main__":
    # 场景 1：需要调 get_weather 工具
    run_function_call("北京明天天气怎么样？")

    # 场景 2：需要调 calculate 工具
    run_function_call("123 乘以 456 等于多少？")

    # 场景 3：不需要任何工具，模型直接回答
    run_function_call("你好，请介绍一下你自己")

    # 场景 4：需要调工具但城市不在 mock 数据中
    run_function_call("成都是什么天气？")

    # 场景 5：多轮连续调用 — 先查两个城市天气，再计算复合运算
    run_function_call("明天北京的温度乘以上海的温度，再除以 3 等于多少？")
