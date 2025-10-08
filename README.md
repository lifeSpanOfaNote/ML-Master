## 新增的 Metrics (Added Metrics)

每次 LLM 调用现在会记录以下指标：

1. **Input Tokens** (输入 token 数)
   - 发送给模型的 prompt 的 token 数量
   - 用于计算成本和监控输入复杂度

2. **Output Tokens** (输出 token 数)
   - 模型生成的响应的 token 数量
   - 用于计算成本和监控输出长度

3. **Total Tokens** (总 token 数)
   - Input + Output tokens
   - 用于总体成本计算

4. **Response Time** (响应时间)
   - 从发送请求到接收完整响应的时间（秒）
   - 用于性能监控和调试

5. **Tokens per Second** (吞吐量)
   - Output tokens / Response time
   - 衡量生成速度的关键指标
   - 对于 streaming 模式特别有用

6. **Model Info** (模型信息) - 仅 OpenAI backend
   - System fingerprint: 模型版本标识
   - Model: 实际使用的模型名称
   - Created: 请求创建时间戳
