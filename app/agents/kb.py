"""知识库问答：一个检索工具 + 一个回答 agent"""

import asyncio
import json
import os

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain.tools import tool
from langchain_community.embeddings import DashScopeEmbeddings
from langgraph.types import Command
from pymilvus import AnnSearchRequest, Function, FunctionType, MilvusClient

from app.agents.llm import ReasoningChatOpenAI
from app.common.logger import logger, setup_logging
from app.db import step_store

load_dotenv()
setup_logging()

MILVUS_URI = os.getenv("MILVUS_URI", "https://milvus-data.langxo.com")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "my_document_collection")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3.7-text-embedding-flash")
RERANK_MODEL = os.getenv("RERANK_MODEL", "gte-rerank-v2")

model = ReasoningChatOpenAI(
    model=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)

embeddings = DashScopeEmbeddings(
    model=EMBED_MODEL,
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
)

milvus = MilvusClient(uri=MILVUS_URI, token=os.getenv("MILVUS_TOKEN"))


class RagState(AgentState):
    pass


def _event(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _shorten(value, limit: int = 1500) -> str:
    text = str(value).replace("\r", " ").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _summarize(value) -> str:
    text = str(value).strip()
    count = text.count("[文档") if "[文档" in text else 0
    if count:
        return f"命中 {count} 段"
    return text.replace("\n", " ")[:60] or "（空）"


@tool
def retrieve_doc(query: str) -> str:
    """根据问题检索知识库，返回带出处的片段"""
    query_vector = embeddings.embed_query(query)

    requests = [
        AnnSearchRequest(data=[query_vector], anns_field="dense", limit=10, param={}),
        AnnSearchRequest(data=[query], anns_field="sparse", limit=10, param={}),
    ]

    reranker = Function(
        name="r1",
        input_field_names=["content"],
        function_type=FunctionType.RERANK,
        params={
            "reranker": "model",
            "provider": "ali",
            "model_name": RERANK_MODEL,
            "queries": [query],
            "max_client_batch_size": 3,
        },
    )

    results = milvus.hybrid_search(
        collection_name=MILVUS_COLLECTION,
        reqs=requests,
        ranker=reranker,
        limit=3,
        output_fields=["content", "doc_name", "clause", "volume", "publish_date"],
    )

    texts = []
    for result in results:
        for res in result:
            entity = res.get("entity", {})
            texts.append(f"[文档: {entity.get('doc_name', '')}]\n{entity.get('content', '')}")
    return "\n\n".join(texts) or "没有检索到相关内容"


@tool
def current_time() -> str:
    """查询当前时间"""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def build_agent(checkpointer):
    return create_agent(
        model=model,
        tools=[retrieve_doc, current_time],
        state_schema=RagState,
        checkpointer=checkpointer,
        system_prompt=(
            "你是一个专业知识库问答助手。用户提问时先调用 retrieve_doc 检索知识库，"
            "只回答检索到的内容，并标出出处；查不到或不确定的，如实说明没有找到。"
        ),
    )


async def chat_stream(agent, message: str, thread_id: str):
    """流式输出：正文 + 思考 + 工具步骤，事件格式和婚礼版保持一致"""
    import time

    config = {"configurable": {"thread_id": thread_id}}
    started: dict[str, float] = {}
    buf: list[str] = []
    tool_calls = 0
    step_log: list[dict] = []
    try:
        async for item in agent.astream(
            {"messages": [HumanMessage(message)]},
            config,
            stream_mode="messages",
            subgraphs=True,
        ):
            namespace, payload = item if len(item) == 2 else ((), item)
            chunk, _metadata = payload

            if isinstance(chunk, AIMessageChunk):
                reasoning = (chunk.additional_kwargs or {}).get("reasoning_content")
                if reasoning:
                    if not namespace:
                        yield _event({"type": "think", "who": "助手", "text": reasoning})
                    else:
                        buf.append(reasoning)
                        if sum(len(p) for p in buf) >= 120:
                            yield _event({"type": "think", "who": "工具", "text": "".join(buf)})
                            buf.clear()

                for call in chunk.tool_call_chunks or []:
                    if call.get("name"):
                        key = call.get("id") or call["name"]
                        started.setdefault(key, time.monotonic())
                        step_log.append({"t": "tool", "who": "助手", "name": call["name"]})
                        yield _event({"type": "tool", "who": "助手", "name": call["name"]})
                    if call.get("args"):
                        yield _event({"type": "args", "frag": call["args"]})

                if chunk.content and not namespace:
                    text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                    yield _event({"type": "text", "text": text})

            elif isinstance(chunk, ToolMessage):
                tool_calls += 1
                name = chunk.name or ""
                elapsed = time.monotonic() - started.pop(chunk.tool_call_id or name, time.monotonic())
                detail = _shorten(chunk.content, 400)
                summary = _summarize(chunk.content)
                step_log.append(
                    {"t": "result", "who": "助手", "name": name, "summary": summary,
                     "detail": detail, "ms": int(elapsed * 1000)}
                )
                yield _event(
                    {"type": "result", "who": "助手", "name": name, "summary": summary,
                     "detail": detail, "ms": int(elapsed * 1000)}
                )

        if buf:
            yield _event({"type": "think", "who": "工具", "text": "".join(buf)})

        try:
            snapshot = await agent.aget_state(config)
            msgs = (snapshot.values or {}).get("messages", [])
            turn_no = sum(1 for m in msgs if isinstance(m, HumanMessage))
            step_store.save(thread_id, turn_no, step_log)
        except Exception as exc:
            logger.warning(f"[步骤存档] 失败: {exc}")

        yield _event({"type": "summary", "tool_calls": tool_calls})
    except Exception as exc:
        logger.exception("对话流中断")
        yield _event({"type": "text", "text": f"\n[执行出错] {type(exc).__name__}: {exc}"})


async def get_messages(checkpointer, thread_id: str) -> list[dict[str, str]]:
    snapshot = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
    if not snapshot:
        return []

    messages = (snapshot.checkpoint.get("channel_values") or {}).get("messages") or []
    result: list[dict[str, str]] = []
    for msg in messages:
        if not msg.content or not isinstance(msg.content, str):
            continue
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            if result and result[-1]["role"] == "assistant":
                result[-1]["content"] += "\n\n" + msg.content
            else:
                result.append({"role": "assistant", "content": msg.content})
    return result


async def clear_messages(checkpointer, thread_id: str):
    await checkpointer.adelete_thread(thread_id)
