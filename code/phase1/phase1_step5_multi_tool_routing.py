"""
Step 5: 多工具路由与条件判断
学习目标：让模型在 4+ 个工具中做正确选择，并根据工具结果做逻辑判断

新增工具：
  - search_product(keyword)：模拟商品搜索
  - convert_currency(amount, from, to)：模拟汇率换算
继承 Step 4 的工具：
  - get_weather(city)
  - calculate(expression)

Milestone 5：
  - 4+ 个工具，选择正确率 >= 80%（10 个测试用例）
  - 意图不匹配时模型直接回复，不强行调 tool
  - 记录错误 case，分析原因
"""

from dotenv import load_dotenv
import os
import json
from openai import OpenAI

load_dotenv()

# ============================================================
# 第 1 步：定义工具函数
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


# ============================================================
# 第 2 步：定义工具的 JSON Schema
# ============================================================

tools = [
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
                    "to_currency": {"type": "string", "description": "目标货币类型，如 USD、CNY、EUR"}
                },
                "required": ["amount", "from_currency", "to_currency"]
            }
        }
    }
]

function_map = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_product": search_product,
    "convert_currency": convert_currency,
}

# ============================================================
# 第 3 步：核心流程 —— 复用 Step 4 的 while 循环模式
# ============================================================

client = OpenAI(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def run_function_call(user_input: str) -> dict:
    """跑通一次完整的 Function Calling，返回选择的工具列表"""
    messages = [{"role": "user", "content": user_input}]
    chosen_tools = []
    round_num = 0

    while True:
        round_num += 1
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            tools=tools,
            temperature=0
        )
        assistant_msg = response.choices[0].message

        if assistant_msg.tool_calls:
            messages.append(assistant_msg)
            for tc in assistant_msg.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)
                chosen_tools.append({"name": func_name, "args": func_args})

                print(f"  -> {func_name}({func_args})")
                func = function_map[func_name]
                result = func(**func_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False)
                })
        else:
            print(f"  [模型直接回复] {assistant_msg.content}")
            break

    return {"chosen_tools": chosen_tools, "rounds": round_num}


# ============================================================
# 第 4 步：Milestone 5 测试 —— 10 个测试用例 + 正确率统计
# ============================================================

# 测试用例：(问题, 期望调用的工具名)
TEST_CASES = [
    # 明确需要工具的场景
    ("北京天气怎么样？", ["get_weather"]),
    ("100美元等于多少人民币？", ["convert_currency"]),
    ("iPhone 现在卖多少钱？", ["search_product"]),
    ("256 除以 8 等于多少？", ["calculate"]),
    ("1000欧元能换多少日元？", ["convert_currency"]),

    # 多工具场景
    ("北京的温度和上海的温度差多少？", ["get_weather", "get_weather", "calculate"]),

    # 意图不匹配、不需要工具的场景
    ("你好，请介绍一下你自己", []),
    ("今天星期几？", []),
    ("帮我写一首关于春天的诗", []),

    # 模糊意图 —— 看模型如何处理
    ("北京到上海的高铁票多少钱？", ["search_product"]),
]


def run_milestone_tests():
    print("=" * 60)
    print("Milestone 5: 多工具路由测试（10 个用例）")
    print("=" * 60)

    correct = 0
    wrong_cases = []

    for i, (question, expected_tools) in enumerate(TEST_CASES, 1):
        expected_names = [t for t in expected_tools]  # 只取工具名
        print(f"\n--- 测试 {i}/{len(TEST_CASES)} ---")
        print(f"  问题: {question}")
        print(f"  期望: {expected_names if expected_names else '不调工具'}")

        result = run_function_call(question)
        chosen_names = [t["name"] for t in result["chosen_tools"]]

        # 判断是否正确：工具名集合一致（忽略顺序，考虑重复次数）
        is_correct = sorted(chosen_names) == sorted(expected_names)
        if is_correct:
            correct += 1
            print(f"  ✅ 正确 (用了 {result['rounds']} 轮)")
        else:
            wrong_cases.append({
                "question": question,
                "expected": expected_names,
                "chosen": chosen_names,
                "rounds": result["rounds"]
            })
            print(f"  ❌ 错误! 期望 {expected_names}, 实际 {chosen_names}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"测试汇总: {correct}/{len(TEST_CASES)} 正确, 正确率 {correct/len(TEST_CASES)*100:.0f}%")
    if wrong_cases:
        print(f"\n错误 Case 分析:")
        for wc in wrong_cases:
            print(f"  问: {wc['question']}")
            print(f"    期望: {wc['expected']} | 实际: {wc['chosen']} ({wc['rounds']} 轮)")
    print(f"{'='*60}")

    return correct >= 8


if __name__ == "__main__":
    success = run_milestone_tests()
    print(f"\n{'✅ Milestone 5 通过!' if success else '❌ 正确率不足 80%，需要分析原因'}")
