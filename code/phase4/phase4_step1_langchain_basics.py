"""
Phase 4 Step 1: LangChain 基础
学习目标：
1. 理解 PromptTemplate、ChatModel、Runnable 三个核心概念
2. 用 LangChain 的 chain 完成一次对话
3. 理解 | 操作符的链式调用（类似 Unix pipe）
"""

import io
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# --- 1. 最基础的 Chain: prompt | llm | parser ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def basic_chain_demo():
    """演示最基础的 LCEL 链式调用"""
    print("=" * 50)
    print("Demo 1: 基础 Chain")
    print("=" * 50)

    # ChatModel: 通过 OpenAI 兼容接口调用 Dashscope 的 qwen-plus
    llm = ChatOpenAI(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    # PromptTemplate: 带变量的 prompt 模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个简洁的助手，回答不超过50字。"),
        ("human", "{question}"),
    ])

    # OutputParser: 提取 LLM 的文本回复
    parser = StrOutputParser()

    # Chain: 用 | 操作符串联，类似 Unix pipe
    # 数据流: dict -> prompt渲染 -> llm调用 -> parser提取 -> 最终字符串
    chain = prompt | llm | parser

    result = chain.invoke({"question": "Python 中列表和元组有什么区别？"})
    print(f"Q: Python 中列表和元组有什么区别？")
    print(f"A: {result}\n")


# --- 2. 管理对话历史 ---
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough


def conversation_history_demo():
    """演示如何用 LangChain 管理多轮对话历史"""
    print("=" * 50)
    print("Demo 2: 对话历史管理")
    print("=" * 50)

    llm = ChatOpenAI(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    # 用 MessagesPlaceholder 动态插入历史消息
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个乐于助人的助手。"),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    # 模拟多轮对话
    chat_history = []
    questions = [
        "我叫小明，我喜欢用 Markdown 格式写东西。",
        "我刚才说了我喜欢用什么格式？",
    ]

    for q in questions:
        print(f"User: {q}")
        # 每次调用时传入当前历史 + 新问题
        result = chain.invoke({"chat_history": chat_history, "question": q})
        print(f"AI: {result}\n")
        # 把本轮对话加入历史
        chat_history.append(HumanMessage(content=q))
        chat_history.append(AIMessage(content=result))


# --- 3. RunnableParallel: 并行执行 ---
from langchain_core.runnables import RunnableParallel


def parallel_demo():
    """演示 RunnableParallel: 同时执行多个 branch"""
    print("=" * 50)
    print("Demo 3: 并行执行 (RunnableParallel)")
    print("=" * 50)

    llm = ChatOpenAI(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    # 对同一个输入，同时生成两种风格的回复
    formal_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "请用正式、专业的语言回答。"),
            ("human", "{question}"),
        ])
        | llm
        | StrOutputParser()
    )

    casual_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "请用轻松、口语化的语言回答。"),
            ("human", "{question}"),
        ])
        | llm
        | StrOutputParser()
    )

    # RunnableParallel 会并行执行两个 branch，结果合并为一个 dict
    parallel = RunnableParallel(formal=formal_chain, casual=casual_chain)

    result = parallel.invoke({"question": "什么是 API？"})
    print("正式风格:", result["formal"])
    print()
    print(" casual 风格:", result["casual"])
    print()


# --- 4. 完整的 CLI 智能助手（用 LangChain 重写 Phase 1 的 assistant）---
def cli_assistant():
    """基于 LangChain 的多轮 CLI 助手"""
    print("=" * 50)
    print("LangChain CLI 助手 (输入 'quit' 退出)")
    print("=" * 50)

    llm = ChatOpenAI(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个乐于助人的 AI 助手，用简洁清晰的语言回答问题。"),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    chat_history = []
    while True:
        user_input = input("\n你: ").strip()
        if not user_input or user_input.lower() == "quit":
            print("再见！")
            break

        result = chain.invoke({"chat_history": chat_history, "question": user_input})
        print(f"AI: {result}")

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=result))


if __name__ == "__main__":
    basic_chain_demo()
    conversation_history_demo()
    parallel_demo()
    # 取消注释可启动交互式 CLI 助手
    # cli_assistant()
