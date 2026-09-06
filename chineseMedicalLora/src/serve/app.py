#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/serve/app.py
================
FastAPI 推理服务入口。

路由:
    GET  /health        健康检查 + 模型可用状态
    POST /chat/stream   SSE 流式问答（微调前 / 微调后可切换）
    /                   静态前端（static/index.html）

启动:
    uvicorn src.serve.app:app --host 0.0.0.0 --port 8000 --workers 1
    或 bash scripts/04_serve.sh

SSE 事件格式（与 static/app.js 约定一致）:
    data: {"token": "片段"}
    data: {"done": true, "tokens": 128, "elapsed": 3.45}
    data: {"error": "错误信息"}
    data: [DONE]
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool

from .inference import model_availability, stream_chat

app = FastAPI(title="chineseMedicalLora 中文医疗问答服务")


# ---- 请求体模型 ----

class ChatMessage(BaseModel):
    role: str = Field(..., description="角色：user / assistant / system")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., description="多轮会话历史")
    model: str = Field(default="merged", pattern="^(merged|base)$",
                       description="merged=微调后, base=微调前")


# ---- 健康检查 ----

@app.get("/health")
def health():
    """服务存活与模型可用状态。模型懒加载，服务启动时即使模型缺失也返回 200。"""
    avail = model_availability()
    finetuned_ready = avail["finetuned_ready"]
    base_ready = avail["base"]["available"]
    return {
        "status": "ok",
        # 任一模型来源可用即视为模型就绪
        "model_loaded": finetuned_ready or base_ready,
        "finetuned_ready": finetuned_ready,
        "base_ready": base_ready,
        "device": avail["device"],
        "models": {
            "merged": avail["merged"],
            "base": avail["base"],
            "lora": avail["lora"],
        },
    }


# ---- 流式问答 ----

def _sse(payload: dict) -> str:
    """构造一条 SSE 事件（中文不转义，按 SSE 规范以空行结尾）。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 流式回答；客户端断开或点「停止」时通过 stop_event 终止生成。"""

    async def event_generator():
        stop_event = threading.Event()
        messages = [m.model_dump() for m in req.messages]
        try:
            sync_iter = stream_chat(
                messages, model_key=req.model, stop_event=stop_event
            )
            # 同步生成器放到线程池迭代，避免阻塞事件循环
            async for event in iterate_in_threadpool(sync_iter):
                kind = event[0]
                if kind == "token":
                    yield _sse({"token": event[1]})
                elif kind == "done":
                    _, tokens, elapsed = event
                    yield _sse({
                        "done": True,
                        "tokens": tokens,
                        "elapsed": round(float(elapsed), 2),
                    })
                    yield "data: [DONE]\n\n"
                elif kind == "error":
                    yield _sse({"error": event[1]})
                    yield "data: [DONE]\n\n"
        except asyncio.CancelledError:
            # 客户端断开（含前端「停止生成」触发的 abort）
            stop_event.set()
            raise
        except Exception as e:
            yield _sse({"error": f"{type(e).__name__}: {e}"})
            yield "data: [DONE]\n\n"
        finally:
            # 兜底：连接结束后确保后台生成线程尽快停止
            stop_event.set()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁止 Nginx 缓冲，保证逐字下发
        },
    )


# ---- 静态前端（必须最后挂载，否则会吞掉 API 路由） ----

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
