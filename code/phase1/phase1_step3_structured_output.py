"""
Phase1 Step3: Structured Output（结构化输出）
学习目标：让LLM输出JSON等机器可读格式，这是后续Function Calling和Agent的基础
"""

from dotenv import load_dotenv
import os
import json
from dashscope import Generation

load_dotenv()
import dashscope
dashscope.api_key = os.getenv("ANTHROPIC_API_KEY")

def basic_json_extraction():
    """基础JSON提取"""
    print("=== 基础JSON提取 ===")
    
    user_input = "张三，28岁，擅长Python和Java，有5年经验。"
    
    messages = [
        {
            "role": "user", 
            "content": f"""
            提取以下信息为JSON：{user_input}
            格式要求：{{"name": "", "age": 0, "skills": [], "experience_years": 0}}
            只输出JSON，不要其他文字。
            """
        }
    ]
    
    response = Generation.call(
        model="qwen-plus",
        messages=messages,
        temperature=0.1  # 低温度保证格式稳定
    )
    
    if response.status_code == 200:
        raw_output = response.output.text.strip()
        print(f"原始输出: {raw_output}")
        
        try:
            # 解析JSON
            parsed_data = json.loads(raw_output)
            print(f"解析成功: {parsed_data}")
            print(f"姓名: {parsed_data['name']}")
            print(f"年龄: {parsed_data['age']}")
            print(f"技能: {parsed_data['skills']}")
            print(f"经验: {parsed_data['experience_years']}")
            
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            return None
    else:
        print(f"API调用失败: {response.code}")
        return None

def complex_json_structures():
    """复杂JSON结构测试"""
    print("\n=== 复杂JSON结构测试 ===")
    
    test_cases = [
        {
            "name": "嵌套对象",
            "input": "公司ABC有技术部和市场部，技术部有10人，市场部有5人。地址在北京朝阳区。",
            "format": {
                "company": {
                    "name": "",
                    "departments": [
                        {"name": "", "employees": 0}
                    ],
                    "address": ""
                }
            }
        },
        {
            "name": "数组结构",
            "input": "产品列表：iPhone 15价格5999，MacBook Pro价格12999，iPad价格4999",
            "format": {
                "products": [
                    {"name": "", "price": 0}
                ]
            }
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- 测试{i}: {case['name']} ---")
        
        messages = [
            {
                "role": "user",
                "content": f"""
                提取信息为JSON：{case['input']}
                格式要求：{json.dumps(case['format'], ensure_ascii=False)}
                只输出JSON，不要其他文字。
                """
            }
        ]
        
        response = Generation.call(
            model="qwen-plus",
            messages=messages,
            temperature=0.1
        )
        
        if response.status_code == 200:
            raw_output = response.output.text.strip()
            print(f"原始输出: {raw_output}")
            
            try:
                parsed_data = json.loads(raw_output)
                print(f"解析成功: {json.dumps(parsed_data, ensure_ascii=False, indent=2)}")
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")

def json_with_error_handling():
    """JSON输出错误处理"""
    print("\n=== JSON输出错误处理 ===")
    
    user_input = "李四，30岁"
    
    for attempt in range(3):
        print(f"\n尝试 {attempt + 1}:")
        
        messages = [
            {
                "role": "user",
                "content": f"""
                提取信息为JSON：{user_input}
                格式要求：{{"name": "", "age": 0}}
                只输出JSON，不要其他文字。
                """
            }
        ]
        
        response = Generation.call(
            model="qwen-plus",
            messages=messages,
            temperature=0.1
        )
        
        if response.status_code == 200:
            raw_output = response.output.text.strip()
            print(f"原始输出: {raw_output}")
            
            try:
                parsed_data = json.loads(raw_output)
                print(f"✅ 解析成功: {parsed_data}")
                return parsed_data
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析失败: {e}")
                
                # 错误处理：重新请求，强调格式
                if attempt < 2:
                    print("重试中...")
                    messages.append({
                        "role": "assistant",
                        "content": raw_output
                    })
                    messages.append({
                        "role": "user",
                        "content": "输出不是有效JSON，请重新输出，只要JSON格式。"
                    })
    
    print("多次尝试后仍失败")
    return None

def batch_structured_extraction():
    """批量结构化提取"""
    print("\n=== 批量结构化提取 ===")
    
    texts = [
        "王五，35岁，前端工程师，会React和Vue",
        "赵六，28岁，后端工程师，会Python和Go",
        "钱七，32岁，全栈工程师，会JavaScript、Python、Docker"
    ]
    
    results = []
    
    for i, text in enumerate(texts, 1):
        print(f"\n处理第{i}条: {text}")
        
        messages = [
            {
                "role": "user",
                "content": f"""
                提取信息为JSON：{text}
                格式要求：{{"name": "", "age": 0, "position": "", "skills": []}}
                只输出JSON，不要其他文字。
                """
            }
        ]
        
        response = Generation.call(
            model="qwen-plus",
            messages=messages,
            temperature=0.1
        )
        
        if response.status_code == 200:
            raw_output = response.output.text.strip()
            
            try:
                parsed_data = json.loads(raw_output)
                results.append(parsed_data)
                print(f"✅ 提取成功: {parsed_data}")
            except json.JSONDecodeError as e:
                print(f"❌ 解析失败: {e}")
                results.append({"error": str(e), "raw": raw_output})
    
    print(f"\n=== 批量提取结果 ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results

def json_schema_validation():
    """JSON Schema验证"""
    print("\n=== JSON Schema验证 ===")
    
    # 定义期望的JSON结构
    expected_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "skills": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["name", "age", "skills"]
    }
    
    user_input = "孙八，25岁，会Python、机器学习、深度学习"
    
    messages = [
        {
            "role": "user",
            "content": f"""
            提取信息为JSON：{user_input}
            格式要求：必须包含name(字符串)、age(数字)、skills(字符串数组)
            示例：{{"name": "张三", "age": 30, "skills": ["Python", "Java"]}}
            只输出JSON，不要其他文字。
            """
        }
    ]
    
    response = Generation.call(
        model="qwen-plus",
        messages=messages,
        temperature=0.1
    )
    
    if response.status_code == 200:
        raw_output = response.output.text.strip()
        print(f"原始输出: {raw_output}")
        
        try:
            parsed_data = json.loads(raw_output)
            
            # 简单验证
            validation_result = validate_json_structure(parsed_data, expected_schema)
            
            if validation_result["valid"]:
                print(f"✅ 验证通过: {parsed_data}")
            else:
                print(f"❌ 验证失败: {validation_result['errors']}")
                
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")

def validate_json_structure(data, schema):
    """简单的JSON结构验证"""
    errors = []
    
    # 检查必需字段
    if "required" in schema:
        for field in schema["required"]:
            if field not in data:
                errors.append(f"缺少必需字段: {field}")
    
    # 检查字段类型
    if "properties" in schema:
        for field, field_schema in schema["properties"].items():
            if field in data:
                expected_type = field_schema["type"]
                actual_type = type(data[field]).__name__
                
                # 类型映射
                type_mapping = {
                    "string": str,
                    "number": (int, float),
                    "array": list,
                    "object": dict
                }
                
                if expected_type in type_mapping:
                    if not isinstance(data[field], type_mapping[expected_type]):
                        errors.append(f"字段{field}类型错误: 期望{expected_type}，实际{actual_type}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

def test_structured_output_success_rate():
    """测试结构化输出成功率"""
    print("\n=== 结构化输出成功率测试 ===")
    
    test_cases = [
        "张三，30岁，程序员",
        "李四，25岁，设计师",
        "王五，28岁，产品经理",
        "赵六，35岁，数据分析师",
        "钱七，32岁，项目经理"
    ]
    
    success_count = 0
    total_count = len(test_cases)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n测试 {i}/{total_count}: {text}")
        
        messages = [
            {
                "role": "user",
                "content": f"""
                提取信息为JSON：{text}
                格式要求：{{"name": "", "age": 0, "profession": ""}}
                只输出JSON，不要其他文字。
                """
            }
        ]
        
        response = Generation.call(
            model="qwen-plus",
            messages=messages,
            temperature=0.1
        )
        
        if response.status_code == 200:
            raw_output = response.output.text.strip()
            
            try:
                json.loads(raw_output)
                print("✅ 成功")
                success_count += 1
            except json.JSONDecodeError:
                print("❌ 失败")
        else:
            print("❌ API调用失败")
    
    success_rate = (success_count / total_count) * 100
    print(f"\n=== 成功率统计 ===")
    print(f"成功: {success_count}/{total_count}")
    print(f"成功率: {success_rate:.1f}%")
    
    return success_rate >= 80

if __name__ == "__main__":
    print("=== Phase1 Step3: 结构化输出 ===")
    
    # 运行所有测试
    print("1. 基础JSON提取")
    basic_json_extraction()
    
    print("\n" + "="*50)
    print("2. 复杂JSON结构测试")
    complex_json_structures()
    
    print("\n" + "="*50)
    print("3. JSON输出错误处理")
    json_with_error_handling()
    
    print("\n" + "="*50)
    print("4. 批量结构化提取")
    batch_structured_extraction()
    
    print("\n" + "="*50)
    print("5. JSON Schema验证")
    json_schema_validation()
    
    print("\n" + "="*50)
    print("6. 结构化输出成功率测试")
    success = test_structured_output_success_rate()
    
    print(f"\n=== Step3 总结 ===")
    if success:
        print("✅ 结构化输出测试通过！")
        print("✅ 连续5次调用，JSON解析成功率 >= 80%")
        print("✅ 能处理模型偶尔输出格式错误的情况")
        print("✅ 了解JSON Schema约束输出的基本用法")
    else:
        print("❌ 成功率不足80%，需要进一步优化")
