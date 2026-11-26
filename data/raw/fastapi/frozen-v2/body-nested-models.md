# Body Nested Models

## 列表字段

Pydantic 模型字段可以声明为 `list[str]`、`set[str]` 等带元素类型的容器。FastAPI 会根据模型定义解析和校验请求体，并在 API 文档中展示字段类型。

## 嵌套模型

一个 Pydantic 模型可以把另一个模型作为字段类型。例如 `Item` 可以包含可选的 `Image`，而 `Offer` 可以包含一组 `Item`。嵌套模型支持编辑器提示、数据转换、校验和自动文档生成。

## 多层请求体

路径参数、query 参数和 Pydantic 请求体可以同时出现在一个路径操作中。FastAPI 会根据参数是否出现在路径、是否是简单类型或是否是 Pydantic 模型，分别从路径、查询字符串或 JSON body 中读取。

