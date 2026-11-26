# Security First Steps

## Bearer token 入门

FastAPI 的安全工具可以从 `Authorization` 请求头读取 Bearer token。缺少该请求头或格式不符合要求时，框架会直接返回 401；符合格式时，路径操作可以收到 token 字符串。

## 这一步不等于完整认证

读取 token 和校验 token 是两个步骤。基础安全依赖只负责提取并检查请求格式，不能自动证明 token 的签名、有效期或用户权限。生产系统还需要接入实际的认证服务和授权规则。

