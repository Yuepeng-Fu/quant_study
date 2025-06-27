### 🛠️ 工具资源：yfinance

#### 📋 基本信息

- **类型**：开源数据源 python API 库
- **官方网站**：[pypi](https://pypi.org/project/yfinance/)
- **文档地址**：[官方文档](https://ranaroussi.github.io/yfinance) 或者 [runoob文档](https://www.runoob.com/python-qt/qt-get-data.html)
- **定价模式**：免费 
- **更新频率**：日频，只能获取

#### 🎯 主要功能与适用场景

- **核心功能**：从 [Yahoo!Ⓡ finance](https://finance.yahoo.com/) 上获取（主要原理是 `https` 请求）开源数据
- **数据覆盖**：~~股票/债券/期货/期权/宏观/另类数据~~ 暂不确定
- **地区覆盖**：全球
- **适用策略类型**：暂不确定

#### 📊 数据质量评估

| 维度     | 评分(1-5) | 说明                  |
| ------ | ------- | ------------------- |
| 数据准确性  | ⭐⭐⭐⭐⭐   | 由yahoo这一大型公司支持，因此   |
| 数据完整性  | ⭐⭐⭐     | 很少，只有开盘、收盘、高点、低点、容量 |
| 实时性    | ⭐⭐      | daily               |
| API稳定性 | ⭐⭐⭐⭐    | 需要vpn才能访问           |
| 文档质量   | ⭐⭐⭐⭐⭐   | 全开源                 |

#### 🔧 技术规格

- **支持语言**：Python
- **数据格式**：~~JSON / CSV / HDF5 / SQL~~
- **API限制**：~~请求频率限制、数据量限制~~
- **认证方式**：~~API Key / Token / OAuth~~

#### 💻 快速上手代码示例

```python
import yfinance as yf

# 上海证券交易所，茅台公司股票代码
symbol = "600519.SS"

# 或者，深圳证券交易所，泸州老窖公司股票代码
# luzhou_laojiao_szse = yf.Ticker('000568.SZ')

# 获取茅台公司股票数据
maotai_data = yf.download(symbol, start="2022-01-01", end="2023-11-01")

# 打印数据的前几行
print(maotai_data.head())
```

#### 💰 成本分析

全部开源

#### ⚠️ 使用注意事项

- **数据延迟**：
- **常见错误**：`YFRateLimitError`，由于yahoo关闭了中国大陆的access，需要通过代理服务器连接。因此在代码中需要设置 `http_proxy` 和 `https_proxy` 环境变量
- **调试技巧**：
- **最佳实践**：

#### 📈 实际应用案例

- **策略1**：[[行业轮动策略]] - 用于获取行业指数数据
- **策略2**：[[因子选股策略]] - 用于财务因子计算
- **研究项目**：[[市场微观结构研究]] - 高频数据分析

#### 🔄 版本更新日志

- last fork: 25-06-26

#### 🤝 社区与支持

- **官方社群**：
- **GitHub仓库**：[yfinance-github](https://github.com/ranaroussi/yfinance)
- **Stack Overflow标签**：
- **中文社区**：

#### 📚 学习资源

- **官方教程**：
- **第三方教程**：
- **视频课程**：
- **相关书籍**：

#### 🔗 相关工具对比

| 工具            | 优势  | 劣势  | 推荐场景 |
| ------------- | --- | --- | ---- |
| yfinance |     |     |      |
| [[替代工具1]]     |     |     |      |
| [[替代工具2]]     |     |     |      |

---

tags:: #工具资源 #数据源 #yfinance