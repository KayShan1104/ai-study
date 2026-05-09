"""
Phase 1 阶段验收：CLI 智能助手

要求：
  1. 支持多轮对话（维护上下文）
  2. 内置 3+ 个工具（至少包含一个计算类、一个查询类）
  3. 模型能正确选择工具或决定不调工具
  4. 工具结果正确传回并用于最终回复
  5. 输出格式为 JSON 时能被正确解析
  6. 打印每次调用的 token 消耗
"""

from dotenv import load_dotenv
import os
import sys
import json
from openai import OpenAI

load_dotenv()

# ============================================================
# 工具定义与执行 —— 与业务逻辑分离
# ============================================================

def get_weather(city: str) -> dict:
    """模拟天气查询"""
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
    """数学计算"""
    try:
        if not all(c in "0123456789+-*/.() " for c in expression):
            return {"error": "表达式包含非法字符"}
        result = eval(expression)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


def search_product(keyword: str) -> dict:
    """模拟商品搜索"""
    mock_data = {
        "iPhone": {"name": "iPhone 15 Pro", "price": 7999, "brand": "Apple"},
        "MacBook": {"name": "MacBook Pro 14寸", "price": 14999, "brand": "Apple"},
        "高铁票": {"name": "北京-上海高铁二等座", "price": 553, "unit": "元/张"},
        "机票": {"name": "北京-上海经济舱", "price": 800, "unit": "元/张"},
        "空调": {"name": "格力空调 1.5匹", "price": 3299, "brand": "格力"},
    }
    for key, product in mock_data.items():
        if key in keyword or keyword in key:
            return {"found": True, **product}
    return {"found": False, "message": f"未找到与「{keyword}」相关的商品"}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """模拟汇率换算"""
    rates = {
        "USD": {"CNY": 7.24, "EUR": 0.92, "JPY": 154.5, "GBP": 0.79},
        "CNY": {"USD": 0.138, "EUR": 0.127, "JPY": 21.34, "GBP": 0.109},
        "EUR": {"USD": 1.087, "CNY": 7.87, "JPY": 168.2, "GBP": 0.859},
        "GBP": {"USD": 1.266, "CNY": 9.17, "EUR": 1.164, "JPY": 195.5},
        "JPY": {"USD": 0.0065, "CNY": 0.047, "EUR": 0.0059, "GBP": 0.0051},
    }
    from_upper = from_currency.upper()
    to_upper = to_currency.upper()
    if from_upper not in rates or to_upper not in rates.get(from_upper, {}):
        return {"error": f"不支持的货币对: {from_upper}/{to_upper}"}
    result = amount * rates[from_upper][to_upper]
    return {"amount": amount, "from": from_upper, "to": to_upper, "result": round(result, 2)}


# 工具定义（JSON Schema，发给 LLM）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市当前的天气状况，包括温度、天气类型和风力。仅用于天气查询。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，如北京、上海"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行纯数学表达式的计算，支持加减乘除和小数运算。不涉及现实世界数据查询。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 1+2*3、100/7"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_product",
            "description": "搜索商品的价格和基本信息。用于查询商品、票务、电子产品等的价格信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "搜索关键词，如 iPhone、高铁票、空调"}
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "货币汇率换算。将指定金额从一种货币转换为另一种货币。不涉及商品价格查询。",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "要转换的金额"},
                    "from_currency": {"type": "string", "description": "源货币类型，如 USD、CNY、EUR"},
                    "to_currency": {"type": "string", "description": "目标货币类型"}
                },
                "required": ["amount", "from_currency", "to_currency"]
            }
        }
    }
]

# 函数名 → 实际函数的映射
FUNCTION_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_product": search_product,
    "convert_currency": convert_currency,
}


# ============================================================
# 核心 Agent 逻辑
# ============================================================

SYSTEM_PROMPT = """你是一个智能助手，可以帮助用户完成多种任务。
可用工具：
- get_weather：查询城市天气
- calculate：数学计算
- search_product：搜索商品价格
- convert_currency：汇率换算

如果用户的问题可以用工具解决，请调用对应工具。
如果不需要工具，请直接回答。
"""


class Assistant:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.round_count = 0

    def _execute_tools(self, tool_calls):
        """执行模型决定的工具调用"""
        results = []
        for tc in tool_calls:
            func_name = tc.function.name
            func_args = json.loads(tc.function.arguments)

            if func_name not in FUNCTION_MAP:
                result = {"error": f"未知工具: {func_name}"}
            else:
                func = FUNCTION_MAP[func_name]
                try:
                    result = func(**func_args)
                except Exception as e:
                    result = {"error": str(e)}

            results.append({
                "tool_call_id": tc.id,
                "name": func_name,
                "args": func_args,
                "result": result
            })

            self.messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False)
            })

        return results

    def _print_usage(self, usage, round_num):
        """打印 token 消耗"""
        if not usage:
            return
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens

        prompt_cost = usage.prompt_tokens / 1000 * 0.0008
        completion_cost = usage.completion_tokens / 1000 * 0.002
        total_cost = prompt_cost + completion_cost

        print(f"\n  [Token #{round_num}] prompt={usage.prompt_tokens}, "
              f"completion={usage.completion_tokens}, total={usage.total_tokens}, "
              f"费用=¥{total_cost:.5f}")

    def chat(self, user_input: str) -> str:
        """处理一轮用户输入"""
        self.round_count += 1
        self.messages.append({"role": "user", "content": user_input})

        while True:
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=self.messages,
                tools=TOOLS,
                temperature=0,
                max_tokens=1024
            )

            self._print_usage(response.usage, self.round_count)

            assistant_msg = response.choices[0].message

            if assistant_msg.tool_calls:
                # 模型决定调工具
                self.messages.append(assistant_msg)
                tool_results = self._execute_tools(assistant_msg.tool_calls)
                print(f"\n  [工具调用] 调用了 {len(tool_results)} 个工具:")
                for tr in tool_results:
                    print(f"    → {tr['name']}({tr['args']}) = {tr['result']}")
            else:
                # 模型直接回复
                return assistant_msg.content


# ============================================================
# CLI 交互入口
# ============================================================

def main():
    print("=" * 60)
    print("CLI 智能助手")
    print("可用工具: get_weather, calculate, search_product, convert_currency")
    print("输入 'quit' 或 'exit' 退出, 输入 'clear' 清空对话")
    print("=" * 60)

    assistant = Assistant()

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        lower = user_input.lower()
        if lower in ("quit", "exit"):
            break

        if lower == "clear":
            assistant.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("[已清空对话历史]")
            continue

        response = assistant.chat(user_input)
        print(f"\n助手: {response}")

    # 打印汇总
    total_rounds = assistant.round_count
    total_tokens = assistant.total_prompt_tokens + assistant.total_completion_tokens
    total_cost = (assistant.total_prompt_tokens / 1000 * 0.0008 +
                  assistant.total_completion_tokens / 1000 * 0.002)

    print(f"\n{'=' * 60}")
    print(f"本次会话总结:")
    print(f"  对话轮数: {total_rounds}")
    print(f"  总 Token 消耗: {total_tokens}")
    print(f"  总费用估算: ¥{total_cost:.5f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
