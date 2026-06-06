# 智能客服系统

基于 **LangGraph 状态图** 和 **RAG（检索增强生成）** 的智能客服系统，支持意图识别、文档检索、回复生成、质量评估与自动改进。

## 项目结构

```
smart_customer_service/
├── main.py              # 入口：日志配置 + 测试运行
├── graph.py             # LangGraph 状态图构建 + 条件路由
├── nodes.py             # 7 个节点函数（意图分析→检索→生成→质检→改进→最终化→历史保存）
├── state.py             # State 类型定义
├── rag.py               # RAG 系统：文档加载、向量化、重排序
├── my_llm.py            # LLM 实例（ChatOpenAI 兼容接口）
├── env_utils.py         # 环境变量加载
├── .env.example         # 环境变量模板
├── requirements.txt     # Python 依赖
└── documents/
    └── product_guide.txt  # RAG 知识库文档
```

## 核心架构

系统采用 **状态图（StateGraph）** 驱动，7 个处理节点通过条件边串联：

```
入口
 │
 ▼
[① 意图分析]───→ chitchat/complaint ──→ [③ 生成回复]
 │                                        │
 │ consultation/return_order              │
 ▼                                        │
[② RAG 检索 + 重排序]                     │
 │                                        │
 └────────────────────────────────────────┘
                   │
                   ▼
            [④ 质量检查]
              │        │
          score<90   score≥90
          & retry<2     │
              │          │
              ▼          ▼
        [⑤ 改进回复]  [⑥ 最终处理]
              │          │
              └─────┐    │
                    │    ▼
                    +→ [④]（重新评分）
                         │
                         ▼
                   [⑥ 最终处理]
                         │
                         ▼
                   [⑦ 保存历史] → END
```

### 节点说明

| 节点 | 文件位置 | 功能 |
|---|---|---|
| `analyze_intent` | `nodes.py` | LLM 将用户输入分类为 4 种意图：`return_order` / `consultation` / `complaint` / `chitchat` |
| `retrieve_documents` | `nodes.py` | 仅咨询/退货类触发。Chroma 向量检索 top-5，经 CrossEncoder 重排序后取 top-2；检索为空则走兜底逻辑 |
| `generate_response` | `nodes.py` | 按意图分三路生成回复：闲聊（友好简短）、投诉（道歉 + 转人工）、咨询/退货（基于 RAG 上下文） |
| `check_quality` | `nodes.py` | 对咨询/退货类回复从准确性、有用性、友好度三个维度打分（0-100） |
| `improve_response` | `nodes.py` | 质量不达标时重试改进，最多 2 次；若改进后退步则自动停止 |
| `finalize` | `nodes.py` | 投诉类追加 `[转人工]` 标记，其余直接输出 |
| `save_history` | `nodes.py` | 保留最近 5 轮对话，注入后续 prompt 实现多轮记忆 |

### State 数据结构

```python
class State(TypedDict):
    user_query: str       # 用户输入
    intent: str           # 意图分类
    need_human: bool      # 是否需要转人工
    rag_context: str      # RAG 检索到的文档片段
    draft_response: str   # 回复草稿（可能被改进）
    quality_score: int    # 质量评分 0-100
    previous_score: int   # 改进前评分（用于退步检测）
    retry_count: int      # 改进重试次数
    final_response: str   # 最终回复
    chat_history: list    # 最近 5 轮对话历史
```

## 技术栈

| 组件 | 技术选型 | 说明 |
|---|---|---|
| LLM | `openai/gpt-oss-20b` (NVIDIA NIM) | 通过 `langchain_openai.ChatOpenAI` 兼容接口调用，可替换为任意 OpenAI 兼容 API |
| Embedding | `bge-large-zh-v1.5` (BAAI) | HuggingFace 本地加载，自动检测 CUDA/CPU |
| Reranker | `bge-reranker-large` (BAAI) | CrossEncoder 重排序，加载失败自动降级 |
| 向量数据库 | Chroma | 本地持久化 |
| 文档加载 | UnstructuredLoader | `langchain_unstructured` |
| 文本切分 | RecursiveCharacterTextSplitter | chunk_size=300, overlap=50, 按段落/句号切分 |
| 编排引擎 | LangGraph | StateGraph 状态机驱动 |

## 快速开始

### 1. 环境要求

- Python 3.10+
- CUDA GPU（可选，CPU 也可运行）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 填入实际值：

```env
API_KEY=your_api_key
BASE_URL=https://integrate.api.nvidia.com/v1
EMBED_MODEL_PATH=/path/to/bge-large-zh-v1.5
RERANKER_MODEL_PATH=/path/to/bge-reranker-large
```

### 4. 准备知识库

编辑 `documents/product_guide.txt`，写入产品帮助文档、FAQ、政策等内容。首次运行时系统会自动完成文档切分和向量化。

### 5. 运行测试

```bash
python main.py
```

系统会依次测试正常咨询、投诉、闲聊以及空输入、超长输入等边界情况。

### 6. API 集成

```python
from graph import build_graph

app = build_graph()

result = app.invoke({
    "user_query": "支持退货吗？",
    "intent": "",
    "need_human": False,
    "rag_context": "",
    "draft_response": "",
    "quality_score": 0,
    "previous_score": 0,
    "retry_count": 0,
    "final_response": "",
    "chat_history": []
})

print(result["final_response"])
```

## 自定义指南

### 添加新的意图类型

1. 在 `nodes.py` 的 `analyze_intent` prompt 中增加新类别
2. 在 `nodes.py` 的 `generate_response` 中增加对应的回复逻辑分支
3. 如需检索，在 `graph.py` 的 `route_by_intent` 中添加路由规则

### 替换 LLM 模型

编辑 `my_llm.py`：

```python
gpt = ChatOpenAI(
    model="your-model-name",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.4
)
```

支持任何 OpenAI 兼容 API。

### 替换 Embedding / Reranker 模型

修改 `.env` 中的 `EMBED_MODEL_PATH` 和 `RERANKER_MODEL_PATH`，指向本地 HuggingFace 模型目录即可。

### 调整检索参数

在 `rag.py` 中修改：

- `chunk_size`：文档切分大小（默认 300）
- `chunk_overlap`：重叠窗口（默认 50）
- `search_kwargs={"k": 5}`：初检召回数量

## 模块依赖关系

```
state  ←  rag  ←  nodes  ←  graph  ←  main
  ↑                  ↑
env_utils ──────────┘
  ↑
my_llm ──────────────┘
```

单向依赖，无循环引用。每个模块可独立测试。

## License

MIT
