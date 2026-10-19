from mcp.server.fastmcp import FastMCP

# 创建 MCP Server
mcp = FastMCP("Python Backend Learning Server")


@mcp.tool()
def add_numbers(a: float, b: float) -> dict:
    """计算两个数字的和。适合处理简单加法问题。"""
    return {
        "a": a,
        "b": b,
        "result": a + b,
    }


@mcp.tool()
def multiply_numbers(a: float, b: float) -> dict:
    """计算两个数字的乘积。"""
    return {
        "a": a,
        "b": b,
        "result": a * b,
    }


@mcp.tool()
def explain_backend_term(term: str) -> dict:
    """解释 Python 后端开发中的常见技术概念。"""

    knowledge = {
        "FastAPI": "Python Web 框架，用于构建 API。",
        "MySQL": "关系型数据库，用于持久化业务数据。",
        "Redis": "内存数据存储，常用于缓存和会话管理。",
        "JWT": "常见的令牌格式，可用于身份认证。",
        "MCP": "让 AI 应用通过标准协议发现并调用工具。",
        "RAG": "先检索相关知识，再把检索结果提供给大模型。",
        "Workflow": "控制任务步骤和执行顺序的工作流。",
    }

    result = knowledge.get(term)

    if result is None:
        return {
            "found": False,
            "term": term,
            "message": "暂时没有这个术语的内置解释。",
        }

    return {
        "found": True,
        "term": term,
        "explanation": result,
    }


# Resource：提供可读取的项目信息
@mcp.resource("project://guide")
def get_project_guide() -> str:
    """返回 Python 后端学习项目的技术栈说明。"""
    return """
    Python 后端学习项目：
    - FastAPI：提供 HTTP API
    - MySQL：保存业务数据
    - Redis：缓存和会话管理
    - JWT：身份认证
    - RAG：检索增强生成
    - MCP：标准化工具和上下文交互
    """


# Prompt：提供可复用的提示模板
@mcp.prompt()
def explain_for_beginner(term: str) -> str:
    """生成面向初学者的技术概念解释提示。"""
    return (
        f"请用中文向 Python 初学者解释 {term}。"
        "请包含概念、用途、简单示例和常见误区。"
    )


if __name__ == "__main__":
    # 本地客户端通过标准输入输出连接
    mcp.run(transport="stdio")
