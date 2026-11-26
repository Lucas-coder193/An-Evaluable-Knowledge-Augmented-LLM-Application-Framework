## HTTPException

在路径操作中需要向客户端报告 HTTP 错误时，可以从 FastAPI 导入并抛出 `HTTPException`。例如查找不到资源时抛出 `HTTPException(status_code=404, detail="Item not found")`，客户端会收到 404 和包含 detail 的 JSON 错误响应。

## 抛出异常而不是返回异常

`HTTPException` 是 Python 异常，因此应使用 `raise`，而不是 `return`。抛出异常会立即终止当前请求处理，即使它发生在路径操作调用的工具函数中，FastAPI 也会将对应 HTTP 错误返回客户端。

## 自定义错误详情

`HTTPException` 的 `detail` 可以是任何能够转换为 JSON 的值，不限于字符串，也可以是字典或列表。构造异常时还可以附加 HTTP 响应头。

## 自定义异常处理

可以使用 `@app.exception_handler()` 为自定义异常注册全局处理函数。FastAPI 也允许覆盖默认异常处理器，包括请求校验异常处理器；`RequestValidationError` 与请求数据校验失败有关，可按需要定制它的响应。
