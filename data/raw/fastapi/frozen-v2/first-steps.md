## 最小应用

一个最简单的 FastAPI 应用先从 `from fastapi import FastAPI` 导入类，再用 `app = FastAPI()` 创建应用实例。使用 `@app.get("/")` 等路径操作装饰器把 URL 路径和 HTTP 方法映射到 Python 函数；函数返回的字典可作为 JSON 响应。

## 自动交互式文档

启动开发服务器后，默认可以在 `/docs` 查看由 Swagger UI 提供的交互式 API 文档，也可以在 `/redoc` 查看另一套文档界面。FastAPI 会根据 API 自动生成 OpenAPI schema，默认可通过 `/openapi.json` 查看原始 JSON 描述。

## OpenAPI 的用途

OpenAPI schema 描述 API 的路径、参数以及输入和输出数据结构。它为自动交互式文档提供依据，也可用于生成与 API 通信的客户端代码，例如前端、移动端或 IoT 客户端。

## 路径操作

路径是 URL 中主机名之后的路由部分，例如 `/items/foo`；操作指 HTTP 方法，例如 GET、POST、PUT 或 DELETE。路径操作装饰器（例如 `@app.get("/")`）把路径和方法关联到其下方的函数。GET 通常用于读取，POST 通常用于创建，PUT 用于更新，DELETE 用于删除；这些是常见约定，FastAPI 不强制每个方法只能用于某个目的。
